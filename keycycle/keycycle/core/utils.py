"""Shared utility functions for keycycle."""

import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Optional, Union

from dotenv import load_dotenv

from ..config.constants import KEY_SUFFIX_LENGTH

# Type alias for key entries: either a string API key or a dict with params
KeyEntry = Union[str, dict[str, Any]]


def normalize_key_entry(
    entry: KeyEntry, api_key_param: str = "api_key"
) -> tuple[str, dict[str, Any]]:
    """
    Normalize a key entry to (primary_key, params_dict).

    Args:
        entry: Either a string API key or a dict containing api_key and other params
        api_key_param: Name of the API key parameter in the dict (default: "api_key")

    Returns:
        Tuple of (primary_key, params_dict) where params_dict always contains the api_key_param

    Raises:
        ValueError: If entry is a dict but doesn't contain api_key_param
        TypeError: If entry is neither str nor dict
    """
    if isinstance(entry, str):
        return entry, {api_key_param: entry}
    if isinstance(entry, dict):
        if api_key_param not in entry:
            raise ValueError(f"Key dict must contain '{api_key_param}'")
        return entry[api_key_param], dict(entry)
    raise TypeError(f"Key entry must be str or dict, got {type(entry).__name__}")


def normalize_key_entries(
    entries: list[KeyEntry], api_key_param: str = "api_key"
) -> list[tuple[str, dict[str, Any]]]:
    """
    Normalize a list of key entries.

    Args:
        entries: List of key entries (strings or dicts)
        api_key_param: Name of the API key parameter

    Returns:
        List of (primary_key, params_dict) tuples
    """
    return [normalize_key_entry(e, api_key_param) for e in entries]


# Rate limit indicators for detection
RATE_LIMIT_INDICATORS: frozenset[str] = frozenset(
    ["429", "too many requests", "rate limit", "resource exhausted", "traffic", "rate-limited"]
)

# Temporary/transient rate limit indicators - retry with SAME key
TEMPORARY_RATE_LIMIT_INDICATORS: frozenset[str] = frozenset(
    [
        "temporarily rate-limited",
        "temporarily unavailable",
        "high traffic",
        "retry shortly",
        "please retry",
        "try again shortly",
        "rate-limited upstream",
        "experiencing high load",
    ]
)

# Hard/quota-based rate limit indicators - rotate to different key
HARD_RATE_LIMIT_INDICATORS: frozenset[str] = frozenset(
    [
        "per-day",
        "per-hour",
        "per-minute",
        "quota exceeded",
        "daily limit",
        "hourly limit",
        "free-models-per-day",
        "x-ratelimit-remaining: 0",
    ]
)

# Auth error indicators
AUTH_STATUS_CODES = {401, 403}
AUTH_INDICATORS: frozenset[str] = frozenset(
    ["401", "403", "unauthorized", "forbidden", "invalid api key", "invalid_api_key", "expired"]
)

# Payment required indicators (e.g. free-tier quota/credits exhausted)
PAYMENT_REQUIRED_STATUS_CODES = {402}
PAYMENT_REQUIRED_INDICATORS: frozenset[str] = frozenset(
    [
        "402",
        "payment required",
        "payment_required",
    ]
)


def get_key_suffix(api_key: str, length: int = KEY_SUFFIX_LENGTH) -> str:
    """
    Extract the suffix of an API key for logging and identification.

    Args:
        api_key: The full API key
        length: Number of characters to extract (default: 8)

    Returns:
        The last `length` characters, or the full key if shorter
    """
    return api_key[-length:] if len(api_key) > length else api_key


def is_rate_limit_error(e: Exception) -> bool:
    """
    Heuristic to detect rate limit errors across different providers.

    Checks:
        1. Exception message for rate limit keywords
        2. status_code attribute for 429
        3. body/response attribute for rate limit indicators

    Args:
        e: The exception to check

    Returns:
        True if this appears to be a rate limit error
    """
    # Check string representation
    err_str = str(e).lower()
    if any(indicator in err_str for indicator in RATE_LIMIT_INDICATORS):
        return True

    # Check status_code attribute
    if hasattr(e, "status_code") and e.status_code == 429:
        return True

    # Check body/response for embedded error info
    body = getattr(e, "body", None) or getattr(e, "response", None)
    if body:
        body_str = str(body).lower()
        if any(indicator in body_str for indicator in RATE_LIMIT_INDICATORS):
            return True

    return False


def is_temporary_rate_limit_error(e: Exception) -> bool:
    """
    Detect if a rate limit error is temporary (upstream congestion)
    vs. a hard quota limit.

    Temporary errors should be retried with the SAME key after a delay.
    Hard limit errors should trigger key rotation.

    Args:
        e: The exception to check

    Returns:
        True if this is a temporary rate limit that should be retried
        with the same key (no rotation needed)
    """
    # First, must be a rate limit error at all
    if not is_rate_limit_error(e):
        return False

    # Build combined error string from all available sources
    err_str = str(e).lower()

    body = getattr(e, "body", None) or getattr(e, "response", None)
    if body:
        err_str += " " + str(body).lower()

    # Check for hard limit indicators first (these take precedence)
    if any(indicator in err_str for indicator in HARD_RATE_LIMIT_INDICATORS):
        return False

    # Check for temporary indicators
    if any(indicator in err_str for indicator in TEMPORARY_RATE_LIMIT_INDICATORS):
        return True

    return False


def is_payment_required_error(e: Exception) -> bool:
    """
    Detect HTTP 402 Payment Required errors (e.g. a Cerebras free-trial key
    whose credits/quota have been exhausted).

    Unlike rate limit errors, a 402 will not clear on its own after a short
    cooldown - the key is exhausted for good (until credits are topped up),
    so callers should treat the key as permanently dead for this process
    rather than retrying it later.

    Args:
        e: The exception to check

    Returns:
        True if this appears to be a payment-required / quota-exhausted error
    """
    # Check status_code attribute
    if hasattr(e, "status_code") and e.status_code in PAYMENT_REQUIRED_STATUS_CODES:
        return True

    # Check string representation
    err_str = str(e).lower()
    if any(indicator in err_str for indicator in PAYMENT_REQUIRED_INDICATORS):
        return True

    # Check body/response for embedded error info
    body = getattr(e, "body", None) or getattr(e, "response", None)
    if body:
        body_str = str(body).lower()
        if any(indicator in body_str for indicator in PAYMENT_REQUIRED_INDICATORS):
            return True

    return False


def is_auth_error(e: Exception) -> bool:
    """
    Detect authentication/authorization errors (invalid/expired keys).

    Args:
        e: The exception to check

    Returns:
        True if this appears to be an auth error (401/403)
    """
    # Check status_code
    if hasattr(e, "status_code") and e.status_code in AUTH_STATUS_CODES:
        return True

    # Check string representation
    err_str = str(e).lower()
    if any(indicator in err_str for indicator in AUTH_INDICATORS):
        return True

    return False


# A key that answers 401/402/403 is unusable for good (revoked, unpaid,
# or its project denied) — no cooldown will fix it, so rotate off it.
DEAD_KEY_STATUS_CODES = AUTH_STATUS_CODES | PAYMENT_REQUIRED_STATUS_CODES


def is_dead_key_error(e: Exception) -> bool:
    """
    Detect errors that mean THIS KEY is dead, not merely busy: HTTP 401/403
    (invalid, revoked, or project-denied key — e.g. Gemini's
    "Your project has been denied access") and 402 (credits exhausted).

    A status_code, when present, is authoritative: 429s are never dead keys
    whatever their message says. Without one, fall back to the string
    heuristics of is_payment_required_error / is_auth_error.
    """
    status = getattr(e, "status_code", None)
    if status in DEAD_KEY_STATUS_CODES:
        return True
    if status == 429:
        return False
    return is_payment_required_error(e) or is_auth_error(e)


def validate_api_key(api_key: str) -> bool:
    """
    Basic validation of API key format.

    Args:
        api_key: The API key to validate

    Returns:
        True if the key appears to be valid format

    Note:
        This only checks format, not whether the key actually works.
    """
    if not api_key or not isinstance(api_key, str):
        return False

    # Most API keys are at least 20 characters
    if len(api_key) < 20:
        return False

    # Check for common placeholder patterns (exact matches or obvious placeholders)
    placeholder_patterns = [
        "your_api_key",
        "placeholder",
        "your-api-key",
        "insert_key",
        "api_key_here",
    ]
    key_lower = api_key.lower()
    if any(pattern in key_lower for pattern in placeholder_patterns):
        return False

    # Check if key is all x's or looks like a template (simplified with regex)
    stripped = re.sub(r"[-_]|sk", "", key_lower)
    if stripped and all(c == "x" for c in stripped):
        return False

    return True


def load_api_keys(
    provider: str,
    env_file: Optional[str] = None,
    extra_params: Optional[list[str]] = None,
    api_key_param: str = "api_key",
) -> list[KeyEntry]:
    """
    Load API keys from environment variables.

    Args:
        provider: Provider name (e.g., 'twelvelabs', 'anthropic')
        env_file: Path to .env file (optional)
        extra_params: List of extra parameter names to load alongside API keys.
            For each param, loads {PROVIDER}_{PARAM}_N environment variables.
        api_key_param: Name for the API key in returned dicts (default: "api_key")

    Returns:
        List of key entries. If extra_params is provided, returns list of dicts
        with api_key and extra params. Otherwise returns list of strings.

    Environment variables format:
        NUM_{PROVIDER}=N
        {PROVIDER}_API_KEY_1, {PROVIDER}_API_KEY_2, ...
        {PROVIDER}_{PARAM}_1, {PROVIDER}_{PARAM}_2, ... (for each extra_param)

    Example:
        >>> # With extra_params=["index_id"], loads:
        >>> # TWELVELABS_API_KEY_1, TWELVELABS_INDEX_ID_1
        >>> # TWELVELABS_API_KEY_2, TWELVELABS_INDEX_ID_2
        >>> keys = load_api_keys("twelvelabs", extra_params=["index_id"])
        >>> # Returns: [{"api_key": "key1", "index_id": "idx1"}, ...]
    """
    if env_file:
        env_path = Path(env_file).resolve()
    else:
        env_path = Path.cwd() / ".env"
    load_dotenv(dotenv_path=env_path, override=True)

    provider_upper = provider.upper()
    num_keys_var = f"NUM_{provider_upper}"
    num_keys_raw = os.getenv(num_keys_var)
    if not num_keys_raw:
        raise ValueError(f"Environment variable '{num_keys_var}' not found.")
    try:
        expected_count = int(num_keys_raw)
    except ValueError:
        raise ValueError(f"'{num_keys_var}' must be an integer, got: {num_keys_raw}")

    # NUM_{PROVIDER} is AUTHORITATIVE (0.3.2, reverting 0.3.1's scan-all):
    # the fleet convention is that the first NUM indices (sorted ascending)
    # are the sanctioned live keys, and keys parked at higher indices beyond
    # a numbering gap are dead or spare on purpose -- loading them would put
    # known-bad keys back into rotation. Gaps INSIDE the sanctioned range
    # are tolerated (a retired key's index can be missing); the loader takes
    # the first NUM indices that actually exist and warns if it comes up
    # short, but never loads past NUM keys.
    key_pattern = re.compile(rf"^{re.escape(provider_upper)}_API_KEY_(\d+)$")
    found: dict[int, str] = {}
    for env_name, value in os.environ.items():
        match = key_pattern.match(env_name)
        if match and value:
            found[int(match.group(1))] = value

    if not found:
        raise ValueError(f"Missing API key: {provider_upper}_API_KEY_1")

    all_indices = sorted(found)
    indices = all_indices[:expected_count]
    excluded = all_indices[expected_count:]
    if excluded:
        logging.getLogger(__name__).info(
            "%s: %s=%d -- using the first %d key indices %s; excluding %d "
            "parked key(s) at indices %s (dead/spare by convention).",
            provider_upper,
            num_keys_var,
            expected_count,
            len(indices),
            indices,
            len(excluded),
            excluded,
        )
    if len(indices) < expected_count:
        logging.getLogger(__name__).warning(
            "%s: only %d API key(s) found (indices %s) but %s=%d -- the "
            "declared count is stale. Using the %d key(s) actually found.",
            provider_upper,
            len(indices),
            indices,
            num_keys_var,
            expected_count,
            len(indices),
        )

    keys: list[KeyEntry] = []
    for i in indices:
        key = found[i]
        if extra_params:
            key_dict: dict[str, Any] = {api_key_param: key}
            for param in extra_params:
                val = os.getenv(f"{provider_upper}_{param.upper()}_{i}")
                if val:
                    key_dict[param] = val
            keys.append(key_dict)
        else:
            keys.append(key)
    return keys


# Type alias for key limit overrides - can be imported from config.dataclasses
KeyLimitOverride = (
    Any  # Actually Union[RateLimits, Dict[str, RateLimits]] but avoiding circular import
)


def normalize_key_limits(
    api_keys: list[str],
    key_limits: Optional[dict[Union[int, str], KeyLimitOverride]],
    logger: Optional[logging.Logger] = None,
) -> dict[str, KeyLimitOverride]:
    """
    Normalize key_limits by converting all identifiers to key suffixes.

    Accepts:
        - Integer index (0-based): Maps to the key at that position
        - String suffix: Matched against the last N characters of each key
        - Full key string: Exact match

    Args:
        api_keys: List of primary API key strings
        key_limits: Dict mapping identifiers to limit overrides
        logger: Optional logger for warnings

    Returns:
        Dict mapping key suffixes to their limit overrides
    """
    if not key_limits:
        return {}

    normalized: dict[str, KeyLimitOverride] = {}

    for identifier, limits in key_limits.items():
        if isinstance(identifier, int):
            # Index-based: get the key at that position
            if 0 <= identifier < len(api_keys):
                suffix = get_key_suffix(api_keys[identifier])
                normalized[suffix] = limits
            elif logger:
                logger.warning(
                    "key_limits index %d is out of range (0-%d)", identifier, len(api_keys) - 1
                )
        elif isinstance(identifier, str):
            # String: try exact match first, then suffix match
            matched = False
            for key in api_keys:
                if key == identifier or key.endswith(identifier):
                    suffix = get_key_suffix(key)
                    normalized[suffix] = limits
                    matched = True
                    break
            if not matched and logger:
                logger.warning(
                    "key_limits identifier '%s' did not match any API key", identifier[-8:]
                )

    return normalized


def resolve_limits(
    model_id: str,
    default_model_id: str,
    key_suffix: Optional[str],
    key_limits: dict[str, KeyLimitOverride],
    model_limits: dict[str, dict[str, Any]],
    provider: str,
    default_limits_factory: Callable[[], Any],
) -> Any:
    """
    Resolve rate limits for a model, optionally with per-key overrides.

    Args:
        model_id: The model identifier
        default_model_id: Fallback model ID if model_id is None
        key_suffix: Optional key suffix for per-key limit lookup
        key_limits: Dict of key suffix -> limit overrides
        model_limits: Dict of provider -> model -> RateLimits (e.g., MODEL_LIMITS)
        provider: Provider name for looking up model limits
        default_limits_factory: Callable that returns default RateLimits

    Returns:
        RateLimits for the model/key combination
    """
    from ..config.dataclasses import RateLimits

    mid = model_id or default_model_id

    # Check key-specific overrides first
    if key_suffix and key_limits:
        override = key_limits.get(key_suffix)
        if override:
            if isinstance(override, RateLimits):
                return override
            elif isinstance(override, dict):
                # Per-model limits for this key
                if mid in override:
                    return override[mid]
                # Check for __default__ fallback
                if "__default__" in override:
                    return override["__default__"]

    # Fall back to provider/model defaults
    provider_limits = model_limits.get(provider, {})
    return provider_limits.get(mid, provider_limits.get("default", default_limits_factory()))
