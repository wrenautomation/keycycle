import pytest

from keycycle import MultiProviderWrapper

# Try importing the actual Agno class if available
try:
    from agno.models.cerebras import Cerebras
    from agno.models.response import ModelResponse
except ImportError:
    Cerebras = None


@pytest.mark.skipif(Cerebras is None, reason="Agno or Cerebras module not installed")
def test_cerebras_direct_access():
    """
    Test that MultiProviderWrapper returns a rotating model that:
    1. Inherits from Agno's Cerebras class.
    2. Can be instantiated and attributes accessed.
    3. Has the rotating mixin functionality.
    """

    try:
        # Initialize the wrapper (loads keys from env)
        wrapper = MultiProviderWrapper.from_env(
            provider="cerebras", default_model_id="qwen-3.8-27b", env_file="./local.env"
        )
    except ValueError as e:
        pytest.skip(f"Skipping: Could not initialize wrapper (missing keys?): {e}")
        return

    # Get the model instance
    model = wrapper.get_model()

    # 1. Verify Inheritance
    assert isinstance(model, Cerebras), "Returned object is not an instance of Cerebras"

    # 2. Verify Attributes
    if hasattr(model, "id"):
        assert model.id == "qwen-3.8-27b"
    elif hasattr(model, "model"):
        assert model.model == "qwen-3.8-27b"

    # 3. Verify Mixin is present
    assert hasattr(model, "wrapper")
    assert model.wrapper is wrapper

    print("Cerebras model instantiated successfully via wrapper.")
    print(f"Class: {type(model).__name__}")

    # 4. Verify Direct Call (REAL)
    prompts = ["What is 2 + 2? Answer in one word.", "Write a haiku about coding."]

    print(f"\n--- Starting Direct API Calls ({len(prompts)} prompts) ---")

    for i, p in enumerate(prompts):
        print(f"\n[Prompt {i + 1}]: {p}")
        try:
            from agno.models.message import Message

            # Direct call to the Agno model
            response: ModelResponse = model.invoke(
                messages=[Message.from_dict({"role": "user", "content": p})],
                assistant_message=Message.from_dict(
                    {"role": "assistant", "content": "You are a helpful assistant."}
                ),
            )

            # Print the result
            print(f"[Response]: {response.content}")

            # Basic validation
            assert response.content, "Response content was empty"

            print(response.to_dict())

        except Exception as e:
            pytest.fail(f"API Call failed for prompt '{p}': {e}")


if __name__ == "__main__":
    if Cerebras is None:
        print("Agno/Cerebras not installed, skipping test.")
    else:
        test_cerebras_direct_access()
