import logging
import os
import time
from threading import RLock
from typing import Any, Callable, Optional, TypeVar, Union

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from .adapters.generic_adapter import (
    AsyncGenericRotatingClient,
    SyncGenericRotatingClient,
    create_rotating_client,
)
from .adapters.openai_adapter import RotatingAsyncOpenAIClient, RotatingOpenAIClient
from .config.constants import DEFAULT_COOLDOWN_SECONDS
from .config.dataclasses import KeyLimitOverride, KeyUsage, RateLimits, UsageSnapshot
from .config.enums import RateLimitStrategy
from .config.log_config import default_logger
from .config.models import MODEL_LIMITS, PROVIDER_STRATEGIES
from .core.backoff import BackoffConfig, ExponentialBackoff
from .core.exceptions import ConfigurationError, KeyNotFoundError, NoAvailableKeyError
from .core.utils import (
    KeyEntry,
    validate_api_key,
)
from .core.utils import (
    load_api_keys as _load_api_keys,
)
from .core.utils import (
    normalize_key_limits as _normalize_key_limits,
)
from .core.utils import (
    resolve_limits as _resolve_limits,
)
from .key_rotation.rotating_mixin import RotatingCredentialsMixin
from .key_rotation.rotation_manager import RotatingKeyManager
from .usage.db_logic import UsageDatabase
from .utils import get_agno_model_class

T = TypeVar("T")


class MultiProviderWrapper:
    """Wrapper for Agno models with rotating API keys"""

    PROVIDER_STRATEGIES = PROVIDER_STRATEGIES
    MODEL_LIMITS = MODEL_LIMITS

    @staticmethod
    def load_api_keys(
        provider: str,
        env_file: Optional[str] = None,
        extra_params: Optional[list[str]] = None,
    ) -> list[KeyEntry]:
        """
        Load API keys from environment variables.

        Args:
            provider: Provider name (e.g., 'twelvelabs', 'anthropic')
            env_file: Path to .env file (optional)
            extra_params: List of extra parameter names to load alongside API keys.
                For each param, loads {PROVIDER}_{PARAM}_N environment variables.

        Returns:
            List of key entries. If extra_params is provided, returns list of dicts
            with api_key and extra params. Otherwise returns list of strings.

        Example:
            >>> # With extra_params=["index_id"], loads:
            >>> # TWELVELABS_API_KEY_1, TWELVELABS_INDEX_ID_1
            >>> # TWELVELABS_API_KEY_2, TWELVELABS_INDEX_ID_2
            >>> keys = MultiProviderWrapper.load_api_keys("twelvelabs", extra_params=["index_id"])
            >>> # Returns: [{"api_key": "key1", "index_id": "idx1"}, ...]
        """
        return _load_api_keys(provider, env_file, extra_params)

    @classmethod
    def from_env(
        cls,
        provider: str,
        default_model_id: str,
        env_file: Optional[str] = None,
        db_url: Optional[str] = None,
        db_env_var: Optional[str] = "TIDB_DB_URL",
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        key_limits: Optional[dict[Union[int, str], KeyLimitOverride]] = None,
        key_tiers: Optional[dict[int, str]] = None,
        tier_limits: Optional[dict[str, RateLimits]] = None,
        extra_params: Optional[list[str]] = None,
        track_usage: bool = True,
        **kwargs,
    ):
        """
        Create a MultiProviderWrapper from environment variables.

        Args:
            provider: Provider name (e.g., 'gemini', 'openai')
            default_model_id: Default model identifier
            env_file: Path to .env file (optional)
            db_url: Database URL for usage persistence
            db_env_var: Environment variable name for database URL
            debug: Enable debug logging
            logger: Custom logger instance
            key_limits: Per-key rate limit overrides (index/suffix -> RateLimits or dict)
            key_tiers: Map key indices to tier names (e.g., {0: 'free', 1: 'pro'})
            tier_limits: Map tier names to rate limits (e.g., {'pro': RateLimits(...)})
            extra_params: List of extra parameter names to load alongside API keys.
                For each param, loads {PROVIDER}_{PARAM}_N environment variables.
            track_usage: Persist per-call usage to the database. When False no
                UsageDatabase is built at all, so no db_url / db_env_var is needed.
            **kwargs: Additional arguments passed to the wrapper

        Example with tiers:
            >>> wrapper = MultiProviderWrapper.from_env(
            ...     provider='gemini',
            ...     default_model_id='gemini-2.5-flash',
            ...     key_tiers={0: 'free', 1: 'pro'},
            ...     tier_limits={
            ...         'free': RateLimits(5, 300, 20, 250000),
            ...         'pro': RateLimits(100, 6000, 1000, 1000000),
            ...     }
            ... )

        Example with extra_params (TwelveLabs with index_id):
            >>> wrapper = MultiProviderWrapper.from_env(
            ...     provider='twelvelabs',
            ...     default_model_id='pegasus-1',
            ...     extra_params=['index_id'],
            ... )
        """
        try:
            model_class = get_agno_model_class(provider)
        except (ValueError, ImportError):
            # Agno might not be installed, or provider not supported by Agno utils.
            # We proceed without a model class, allowing only OpenAI client usage.
            model_class = None

        api_keys = cls.load_api_keys(provider, env_file, extra_params)
        if track_usage:
            db_url = db_url or os.getenv(db_env_var)

        # Convert tier config to key_limits
        if key_tiers and tier_limits:
            key_limits = key_limits or {}
            for key_index, tier_name in key_tiers.items():
                if tier_name in tier_limits:
                    key_limits[key_index] = tier_limits[tier_name]

        return cls(
            provider,
            api_keys,
            default_model_id,
            model_class,
            db_url,
            db_env_var,
            debug,
            logger,
            key_limits=key_limits,
            track_usage=track_usage,
            **kwargs,
        )

    def __init__(
        self,
        provider: str,
        api_keys: list[KeyEntry],
        default_model_id: str,
        model_class: Optional[Any] = None,
        db_url: Optional[str] = None,
        db_env_var: Optional[str] = "TIDB_DB_URL",
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
        key_limits: Optional[dict[Union[int, str], KeyLimitOverride]] = None,
        track_usage: bool = True,
        **kwargs,
    ):
        self.provider = provider.lower()
        self.logger = logger or default_logger
        self.model_class = model_class
        self.default_model_id = default_model_id
        self.model_kwargs = kwargs
        self.cooldown_seconds = cooldown_seconds
        self.toggle_debug(debug)

        # Extract primary keys for validation (handle both str and dict entries)
        primary_keys = [
            entry if isinstance(entry, str) else entry.get("api_key", "") for entry in api_keys
        ]

        # Validate API keys
        for i, key in enumerate(primary_keys):
            if not validate_api_key(key):
                self.logger.warning("API key #%d appears invalid or is a placeholder", i + 1)

        # Normalize key_limits: convert key identifiers (index/suffix/full key) to suffixes
        self._key_limits: dict[str, KeyLimitOverride] = self._normalize_key_limits_internal(
            primary_keys, key_limits
        )

        # With tracking off there is nothing to persist, so no database is built
        # and no db_url / TIDB_DB_URL is required.
        self._track_usage = bool(track_usage)
        self.db = UsageDatabase(db_url, db_env_var) if self._track_usage else None
        self.strategy = self.PROVIDER_STRATEGIES.get(self.provider, RateLimitStrategy.PER_MODEL)
        self.manager = RotatingKeyManager(
            api_keys,
            self.provider,
            self.strategy,
            self.db,
            cooldown_seconds=cooldown_seconds,
            limit_resolver=self._resolve_limits_internal,
            track_usage=self._track_usage,
        )
        self._model_cache_lock = RLock()  # Thread safety for RotatingClass creation
        self._RotatingClass = None
        self.console = Console()

    @property
    def track_usage(self) -> bool:
        """Whether per-call usage is written to the usage database."""
        return self._track_usage

    @track_usage.setter
    def track_usage(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled and self.db is None:
            raise ConfigurationError(
                "usage tracking was requested but no usage database is configured: "
                f"this wrapper for '{self.provider}' was built with track_usage=False. "
                "Rebuild it with track_usage=True (and a db_url) to persist usage."
            )
        self._track_usage = enabled
        self.manager.track_usage = enabled

    def toggle_debug(self, enable: bool) -> None:
        """
        Dynamically switches logging verbosity for this module.
        enable=True  -> Shows detailed rotation/reservation logs (DEBUG)
        enable=False -> Shows only key info/warnings (INFO)
        """
        level = logging.DEBUG if enable else logging.INFO
        self.logger.setLevel(level)

        status = "ENABLED" if enable else "DISABLED"
        self.logger.info("Debug logging %s for %s", status, self.provider)

    def _normalize_key_limits_internal(
        self, api_keys: list[str], key_limits: Optional[dict[Union[int, str], KeyLimitOverride]]
    ) -> dict[str, KeyLimitOverride]:
        """
        Normalize key_limits by converting all identifiers to key suffixes.

        Accepts:
            - Integer index (0-based): Maps to the key at that position
            - String suffix: Matched against the last N characters of each key
            - Full key string: Exact match

        Returns:
            Dict mapping key suffixes to their limit overrides
        """
        return _normalize_key_limits(api_keys, key_limits, self.logger)

    def _resolve_limits_internal(
        self, model_id: str, key_suffix: Optional[str] = None
    ) -> RateLimits:
        """
        Resolve rate limits for a model, optionally with per-key overrides.

        Args:
            model_id: The model identifier
            key_suffix: Optional key suffix for per-key limit lookup

        Returns:
            RateLimits for the model/key combination
        """
        return _resolve_limits(
            model_id=model_id,
            default_model_id=self.default_model_id,
            key_suffix=key_suffix,
            key_limits=self._key_limits,
            model_limits=self.MODEL_LIMITS,
            provider=self.provider,
            default_limits_factory=lambda: RateLimits(10, 100, 1000),
        )

    def get_key_usage(
        self,
        model_id: str = None,
        estimated_tokens: int = 1000,
        wait: bool = True,
        timeout: float = 10,
        key_id: Union[int, str] = None,
    ) -> KeyUsage:
        """
        Finds the first valid key, or a specific key if key_id is provided.

        Args:
            model_id: Model identifier
            estimated_tokens: Estimated token usage
            wait: Whether to wait for a key if none available (ignored if key_id is set)
            timeout: Max wait time
            key_id: Optional index (int) or suffix/key (str) to force a specific key

        Returns:
            KeyUsage object for the selected key

        Raises:
            KeyNotFoundError: If key_id is specified but not found
            NoAvailableKeyError: If no keys are available within timeout
        """
        mid = model_id or self.default_model_id

        # Specific Key Request (Bypass Rotation Logic)
        if key_id is not None:
            key_usage = self.manager.get_specific_key(key_id, mid, estimated_tokens)
            if not key_usage:
                raise KeyNotFoundError(key_id)
            return key_usage

        # Standard Rotation Logic with Exponential Backoff
        limits = self._resolve_limits_internal(mid)
        start = time.time()
        backoff = ExponentialBackoff(
            BackoffConfig(
                initial_interval=0.5,
                max_interval=2.0,  # Cap at 2 seconds for key polling
                multiplier=1.5,
            )
        )

        while True:
            key_usage = self.manager.get_key(mid, limits, estimated_tokens)
            if key_usage:
                return key_usage
            if not wait:
                raise NoAvailableKeyError(
                    self.provider,
                    mid,
                    wait=False,
                    timeout=timeout,
                    total_keys=len(self.manager.keys),
                )
            if time.time() - start > timeout:
                # Count cooling down keys for better error message
                cooling_down = sum(
                    1 for k in self.manager.keys if k.is_cooling_down(self.cooldown_seconds)
                )
                raise NoAvailableKeyError(
                    self.provider,
                    mid,
                    wait=True,
                    timeout=timeout,
                    total_keys=len(self.manager.keys),
                    cooling_down=cooling_down,
                )
            backoff.wait()

    def get_openai_client(
        self, estimated_tokens: int = 1000, max_retries: int = 5, **kwargs
    ) -> RotatingOpenAIClient:
        """
        Returns a rotating OpenAI client (Sync)

        Args:
            estimated_tokens: Estimated tokens per request for rate limiting
            max_retries: Maximum retries on rate limit errors
            **kwargs: Additional arguments passed to the OpenAI client
        """
        return RotatingOpenAIClient(
            manager=self.manager,
            limit_resolver=self._resolve_limits_internal,
            default_model=self.default_model_id,
            estimated_tokens=estimated_tokens,
            max_retries=max_retries,
            provider=self.provider,  # Pass provider so it can look up base_url
            client_kwargs={**self.model_kwargs, **kwargs},
        )

    def get_async_openai_client(
        self, estimated_tokens: int = 1000, max_retries: int = 5, **kwargs
    ) -> RotatingAsyncOpenAIClient:
        """
        Returns a rotating OpenAI client (Async)

        Args:
            estimated_tokens: Estimated tokens per request for rate limiting
            max_retries: Maximum retries on rate limit errors
            **kwargs: Additional arguments passed to the AsyncOpenAI client
        """
        return RotatingAsyncOpenAIClient(
            manager=self.manager,
            limit_resolver=self._resolve_limits_internal,
            default_model=self.default_model_id,
            estimated_tokens=estimated_tokens,
            max_retries=max_retries,
            provider=self.provider,  # Pass provider so it can look up base_url
            client_kwargs={**self.model_kwargs, **kwargs},
        )

    def get_rotating_client(
        self,
        client_class: type[T],
        api_key_param: str = "api_key",
        is_async: Optional[bool] = None,
        usage_extractor: Optional[Callable[[Any], int]] = None,
        estimated_tokens: int = 1000,
        max_retries: int = 5,
        model_param: str = "model",
        excluded_kwargs: Optional[list[str]] = None,
        **client_kwargs,
    ) -> Union[SyncGenericRotatingClient[T], AsyncGenericRotatingClient[T]]:
        """
        Returns a rotating client that wraps any Python client class accepting api_key=.

        This creates a drop-in replacement for the original client class that automatically
        rotates API keys on rate limit errors.

        Args:
            client_class: The client class to wrap (e.g., Anthropic, TwelveLabs)
            api_key_param: Name of the API key parameter in the client constructor
            is_async: Whether the client is async. If None, auto-detected
            usage_extractor: Custom function to extract token usage from responses.
                If None, uses a default extractor that handles OpenAI, Anthropic,
                and Cohere response patterns
            estimated_tokens: Estimated tokens per request for rate limiting
            max_retries: Maximum number of key rotations on rate limit errors
            model_param: Name of the model parameter in API calls
            excluded_kwargs: List of kwarg names to exclude from client constructor.
                Useful for clients that don't accept certain params (e.g., 'model' for TwelveLabs).
            **client_kwargs: Additional kwargs to pass to the client constructor

        Returns:
            A rotating client wrapper that can be used like the original client

        Examples:
            >>> # Anthropic
            >>> from anthropic import Anthropic, AsyncAnthropic
            >>> wrapper = MultiProviderWrapper.from_env(provider="anthropic", default_model_id="claude-3-sonnet")
            >>> client = wrapper.get_rotating_client(Anthropic)
            >>> response = client.messages.create(model="claude-3-sonnet", messages=[...], max_tokens=100)

            >>> # Async variant
            >>> async_client = wrapper.get_rotating_client(AsyncAnthropic)
            >>> response = await async_client.messages.create(...)

            >>> # TwelveLabs with excluded_kwargs (doesn't accept 'model')
            >>> from twelvelabs import TwelveLabs
            >>> wrapper = MultiProviderWrapper.from_env(provider="twelvelabs", default_model_id="marengo2.6")
            >>> client = wrapper.get_rotating_client(TwelveLabs, excluded_kwargs=["model"])
            >>> indexes = client.index.list()

            >>> # Custom usage extractor for Cohere
            >>> client = wrapper.get_rotating_client(
            ...     cohere.Client,
            ...     usage_extractor=lambda r: (
            ...         (r.meta.billed_units.input_tokens or 0) +
            ...         (r.meta.billed_units.output_tokens or 0)
            ...     )
            ... )
        """
        return create_rotating_client(
            client_class=client_class,
            manager=self.manager,
            limit_resolver=self._resolve_limits_internal,
            default_model=self.default_model_id,
            api_key_param=api_key_param,
            is_async=is_async,
            usage_extractor=usage_extractor,
            estimated_tokens=estimated_tokens,
            max_retries=max_retries,
            model_param=model_param,
            excluded_kwargs=excluded_kwargs,
            **{**self.model_kwargs, **client_kwargs},
        )

    def get_model(
        self,
        estimated_tokens: int = 1000,
        wait: bool = True,
        timeout: float = 10,
        max_retries: int = 5,
        key_id: Union[int, str] = None,
        pin_key: bool = False,
        **kwargs,
    ) -> Any:
        """Dynamically creates a rotating model for ANY provider."""
        if self.model_class is None:
            raise RuntimeError(
                "Agno model class is not available. "
                "Ensure 'agno' is installed and that the provider is supported."
            )

        # Thread-safe creation of rotating class
        with self._model_cache_lock:
            if self._RotatingClass is None:
                self._RotatingClass = type(
                    f"Rotating{self.model_class.__name__}",
                    (RotatingCredentialsMixin, self.model_class),
                    {},
                )
        RotatingProviderClass = self._RotatingClass

        model_id = kwargs.get("id", self.default_model_id)
        final_kwargs = {**self.model_kwargs, **kwargs}
        if "id" not in final_kwargs:
            final_kwargs["id"] = model_id

        initial_key_usage = self.get_key_usage(
            model_id=model_id,
            estimated_tokens=estimated_tokens,
            wait=wait,
            timeout=timeout,
            key_id=key_id,
        )

        fixed_key_id = key_id if pin_key else None

        model_instance = RotatingProviderClass(
            api_key=initial_key_usage.api_key,
            model_id=model_id,
            wrapper=self,
            rotating_wait=wait,
            rotating_timeout=timeout,
            rotating_estimated_tokens=estimated_tokens,
            rotating_max_retries=max_retries,
            rotating_fixed_key_id=fixed_key_id,
            **final_kwargs,
        )

        return model_instance

    def get_api_key(
        self,
        model_id: Optional[str] = None,
        estimated_tokens: int = 1000,
        wait: bool = True,
        timeout: float = 10,
        key_id: Union[int, str] = None,
    ) -> str:
        """
        Get a valid API key for direct use (e.g., embeddings, custom endpoints).

        Args:
            model_id: Model identifier for rate limiting (uses default if None)
            estimated_tokens: Estimated tokens for this request
            wait: Whether to wait for an available key
            timeout: Maximum time to wait for a key
            key_id: Optional index (int) or suffix/key (str) to force a specific key

        Returns:
            A valid API key string

        Example:
            >>> wrapper = MultiProviderWrapper.from_env('cohere', 'command-r-plus')
            >>> api_key = wrapper.get_api_key()
            >>> # Use with cohere SDK directly
            >>> import cohere
            >>> co = cohere.Client(api_key)
            >>> response = co.embed(texts=["hello"], model="embed-english-v3.0")
        """
        key_usage = self.get_key_usage(model_id, estimated_tokens, wait, timeout, key_id=key_id)
        return key_usage.api_key

    def get_api_key_with_context(
        self,
        model_id: Optional[str] = None,
        estimated_tokens: int = 1000,
        wait: bool = True,
        timeout: float = 10,
        key_id: Union[int, str] = None,
    ) -> tuple[str, KeyUsage]:
        """
        Get an API key along with its usage context for manual tracking.
        This gives you both the key and the key_usage object for more control.

        Args:
            model_id: Model identifier for rate limiting
            estimated_tokens: Estimated tokens for this request
            wait: Whether to wait for an available key
            timeout: Maximum time to wait
            key_id: Optional index (int) or suffix/key (str) to force a specific key

        Returns:
            Tuple of (api_key: str, key_usage_obj: KeyUsage)
        """
        key_usage = self.get_key_usage(model_id, estimated_tokens, wait, timeout, key_id=key_id)
        return key_usage.api_key, key_usage

    def record_key_usage(
        self,
        api_key: str,
        model_id: Optional[str] = None,
        actual_tokens: int = 0,
        estimated_tokens: int = 1000,
    ) -> None:
        """
        Record usage for a key obtained via get_api_key().
        Call this after you're done using the key to update usage tracking.

        Args:
            api_key: The API key that was used
            model_id: Model that was used (uses default if None)
            actual_tokens: Actual tokens consumed (if known)
            estimated_tokens: Estimated tokens (used if actual unknown)

        Example:
            >>> api_key = wrapper.get_api_key(estimated_tokens=500)
            >>> # ... use api_key for embeddings ...
            >>> wrapper.record_key_usage(api_key, model_id="embed-english-v3.0", actual_tokens=450)
        """
        mid = model_id or self.default_model_id

        # Find the key_usage object for this api_key
        key_obj = next((k for k in self.manager.keys if k.api_key == api_key), None)
        if key_obj:
            self.manager.record_usage(key_obj, mid, actual_tokens, estimated_tokens)
        else:
            self.logger.warning("API key not found in manager for recording usage")

    # --- PRINTING HELPERS ---

    def _create_usage_table(self, title: str, data: list[tuple[str, UsageSnapshot]]) -> Table:
        """
        Generates a standardized table for usage stats.
        data format: [(Label, Snapshot), ...]
        """
        # Palette
        c_title = "#bae1ff"  # Pastel Rose
        c_head = "#f2f2f2"  # Pastel Cream
        c_req = "#faa0a0"  # Pastel Periwinkle
        c_tok = "#e5baff"  # Pastel Peach
        c_border = "#B9B9B9"  # Muted Grey
        c_identifier = "#7cd292"  # Soft Mauve

        table = Table(
            title=title,
            box=box.ROUNDED,
            expand=False,
            title_style=f"bold {c_title}",
            title_justify="left",
            border_style=c_border,
            header_style=f"{c_head}",
        )

        # Define Columns
        table.add_column("Identifier", style=f"bold {c_identifier}", no_wrap=True)
        table.add_column(
            "Requests ([white]m / h / d[/])", justify="center", style=c_req, no_wrap=True
        )
        table.add_column(
            "Tokens ([white]m / h / d[/])", justify="center", style=c_tok, no_wrap=True
        )
        table.add_column("Total Requests", justify="center", style=f"{c_req}", no_wrap=True)
        table.add_column("Total Tokens", justify="center", style=f"bold {c_tok}", no_wrap=True)

        for label, s in data:
            req_str = f"{s.rpm} / {s.rph} / {s.rpd}"
            tok_str = f"{s.tpm:,} / {s.tph:,} / {s.tpd:,}"

            table.add_row(label, req_str, tok_str, f"{s.total_requests}", f"{s.total_tokens:,}")
        return table

    def print_global_stats(self) -> None:
        stats = self.manager.get_global_stats()

        # 1. Prepare Data for the Table
        rows = []
        for k in stats.keys:
            label = f"Key #{k.index + 1} (..{k.suffix})"
            rows.append((label, k.snapshot))

        # 2. Create and Print Table
        table = self._create_usage_table(
            title=f"GLOBAL STATS: {escape(self.provider.upper())}", data=rows
        )

        # 3. Add a Summary Footer (using a Panel for the Total)
        total_s = stats.total
        grid = Table.grid(padding=(0, 4))
        grid.add_column(style="#e0e0e0")  # Label Color
        grid.add_column(style="bold", justify="left")  # Value Color

        grid.add_row("Total Requests:", f"[{'#faa0a0'}]{total_s.total_requests}[/]")
        grid.add_row("Total Tokens:", f"[{'#e5baff'}]{total_s.total_tokens:,}[/]")

        self.console.print()
        self.console.print(
            Panel(
                grid,
                title="[bold #bae1ff] AGGREGATE TOTALS [/]",
                border_style="#bae1ff",
                expand=False,
            )
        )
        self.console.print(table)

    def print_key_stats(self, identifier: Union[int, str]) -> None:
        stats = self.manager.get_key_stats(identifier)
        if not stats:
            self.console.print(f"[bold red]Key not found:[/][white] {identifier}[/]")
            return

        self.console.print()
        self.console.rule(f"[bold]Key Report: {stats.suffix}[/]")

        # 1. Total Snapshot Panel
        s = stats.total
        grid = Table.grid(padding=(0, 4))
        grid.add_column(style="#e0e0e0")
        grid.add_column(justify="left")

        grid.add_row("Total Requests:", f"[{'#faa0a0'}]{s.total_requests}[/]")
        grid.add_row("Total Tokens:", f"[{'#e5baff'}]{s.total_tokens:,}[/]")

        self.console.print(
            Panel(
                grid,
                title=f"[bold #97e3e9]Key #{stats.index + 1} Overview[/]",
                border_style="#bae1ff",
                expand=False,
            )
        )

        # 2. Breakdown Table
        if not stats.breakdown:
            self.console.print("[italic dim]No usage recorded for this key yet.[/]")
        else:
            rows = [(model_id, snap) for model_id, snap in stats.breakdown.items()]
            table = self._create_usage_table(title="Breakdown by Model", data=rows)
            self.console.print(table)

    def print_model_stats(self, model_id: str) -> None:
        data = self.manager.get_model_stats(model_id)

        self.console.print()
        self.console.rule(f"[bold]Model Report: [blue]{model_id}[/][/]", style="#B9B9B9")

        # 1. Total Summary
        s = data.total
        self.console.print(f"Total Tokens Consumed: [bold green]{s.total_tokens:,}[/]")

        # 2. Contributing Keys Table
        if not data.keys:
            self.console.print("[italic dim]No keys have used this model.[/]")
        else:
            rows = []
            for k in data.keys:
                label = f"Key #{k.index + 1} (..{k.suffix})"
                rows.append((label, k.snapshot))

            table = self._create_usage_table(title="Contributing Keys", data=rows)
            self.console.print(table)

    def print_granular_stats(self, identifier: Union[int, str], model_id: str) -> None:
        data = self.manager.get_granular_stats(identifier, model_id)

        if not data:
            self.console.print(f"[bold red]Key '{identifier}' not found.[/]")
            return

        self.console.print()
        if data.snapshot:
            # Re-use the table builder for a single row just for consistency
            label = f"Key #{data.index + 1} (..{data.suffix})"
            table = self._create_usage_table(
                title=f"Granular: {model_id}", data=[(label, data.snapshot)]
            )
            self.console.print(table)
        else:
            self.console.print(
                Panel(
                    f"No usage for model [bold]{model_id}[/] on key [bold]..{data.suffix}[/]",
                    style="#e5baff",
                    border_style="#B9B9B9",
                )
            )
