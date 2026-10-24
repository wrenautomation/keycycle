import importlib


def get_agno_model_class(provider: str):
    """
    Dynamically maps a provider string to the actual Agno model class.
    """
    p_low = provider.lower()

    overrides = {
        "openai": "OpenAI",
        "google": "Gemini",
        "gemini": "Gemini",
        "azure": "AzureOpenAI",
        "aws": "Bedrock",
        "openrouter": "OpenRouter",
        "moonshot": "MoonShot",
    }

    class_name = overrides.get(p_low, p_low.capitalize())

    module_path = "google" if p_low in ["google", "gemini"] else p_low

    try:
        module = importlib.import_module(f"agno.models.{module_path}")
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Agno class '{class_name}' not found for provider '{provider}': {e}")
