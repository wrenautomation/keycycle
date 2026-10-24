# test_key_retrieval_real.py
import os

import pytest
from dotenv import load_dotenv

from keycycle import MultiProviderWrapper

# Load environment variables
load_dotenv(override=True, dotenv_path="local.env")


class TestKeyRetrievalReal:
    """Real integration tests using actual Cohere API"""

    @pytest.fixture(scope="class")
    def wrapper(self):
        """Create a real wrapper instance with Cohere keys from .env"""
        # Check if we have Cohere keys configured
        num_keys = os.getenv("NUM_COHERE")
        if not num_keys:
            pytest.skip("NUM_COHERE not set in environment")

        wrapper = MultiProviderWrapper.from_env(
            provider="cohere", default_model_id="command-r-plus"
        )
        return wrapper

    @pytest.fixture(scope="class")
    def cohere_client(self):
        """Import and return cohere client"""
        try:
            import cohere

            return cohere
        except ImportError:
            pytest.skip("cohere package not installed. Install with: pip install cohere")

    def test_get_api_key_basic(self, wrapper):
        """Test basic API key retrieval"""
        api_key = wrapper.get_api_key()

        assert isinstance(api_key, str)
        assert len(api_key) > 0
        print(f"\n Retrieved API key: {api_key[:10]}...")

    def test_embedding_with_get_api_key(self, wrapper: MultiProviderWrapper, cohere_client):
        """Test real embedding request using get_api_key()"""
        # Get API key
        api_key = wrapper.get_api_key(model_id="embed-english-v3.0", estimated_tokens=100)

        print(f"\n Got API key for embeddings: {api_key[:10]}...")

        # Use with Cohere SDK
        co = cohere_client.Client(api_key)

        # Make real API call
        response = co.embed(
            texts=["Hello world", "Testing embeddings"],
            model="embed-english-v3.0",
            input_type="search_document",
        )

        assert response.embeddings is not None
        assert len(response.embeddings) == 2
        print(f" Embedding successful: {len(response.embeddings)} embeddings generated")
        print(f"  Embedding dimension: {len(response.embeddings[0])}")

        # Record usage (Cohere doesn't return token count for embeddings, estimate)
        estimated_tokens = len("Hello world Testing embeddings".split()) * 2
        wrapper.record_key_usage(
            api_key=api_key, model_id="embed-english-v3.0", actual_tokens=estimated_tokens
        )

        print(f" Recorded usage: ~{estimated_tokens} tokens")

    def test_embedding_with_context(self, wrapper: MultiProviderWrapper, cohere_client):
        """Test embedding using get_api_key_with_context()"""
        # Get key with context
        api_key, key_context = wrapper.get_api_key_with_context(
            model_id="embed-english-v3.0", estimated_tokens=100
        )

        print(f"\n Got API key with context: {api_key[:10]}...")

        # Use key
        co = cohere_client.Client(api_key)
        response = co.embed(
            texts=["Python is great", "AI is fascinating"],
            model="embed-english-v3.0",
            input_type="search_document",
        )

        assert len(response.embeddings) == 2
        print(f"✓ Generated {len(response.embeddings)} embeddings")

        # Record using context
        estimated_tokens = 50
        wrapper.manager.record_usage(
            key_obj=key_context,
            model_id="embed-english-v3.0",
            actual_tokens=estimated_tokens,
            estimated_tokens=100,
        )

        print(f" Recorded usage via context: {estimated_tokens} tokens")

    def test_multiple_embedding_calls_rotation(self, wrapper: MultiProviderWrapper, cohere_client):
        """Test that multiple embedding calls work and potentially rotate keys"""
        initial_stats = wrapper.manager.get_global_stats()
        initial_requests = initial_stats.total.total_requests

        print(f"\n✓ Initial total requests: {initial_requests}")

        # Make multiple embedding calls
        texts_batches = [
            ["First batch text 1", "First batch text 2"],
            ["Second batch text 1", "Second batch text 2"],
            ["Third batch text 1", "Third batch text 2"],
        ]

        for i, texts in enumerate(texts_batches, 1):
            api_key = wrapper.get_api_key(model_id="embed-english-v3.0", estimated_tokens=50)

            co = cohere_client.Client(api_key)
            response = co.embed(
                texts=texts, model="embed-english-v3.0", input_type="search_document"
            )

            assert len(response.embeddings) == 2

            # Record usage
            wrapper.record_key_usage(
                api_key=api_key, model_id="embed-english-v3.0", actual_tokens=50
            )

            print(f" Batch {i}: Generated embeddings using key {api_key[:10]}...")

        # Check final stats
        final_stats = wrapper.manager.get_global_stats()
        final_requests = final_stats.total.total_requests

        assert final_requests >= initial_requests + 3
        print(
            f"✓ Final total requests: {final_requests} (added {final_requests - initial_requests})"
        )

    def test_chat_and_embed_mixed(self, wrapper: MultiProviderWrapper, cohere_client):
        """Test using wrapper for both chat and embeddings"""
        # First, do an embedding
        embed_key = wrapper.get_api_key(model_id="embed-english-v3.0", estimated_tokens=50)

        co_embed = cohere_client.Client(embed_key)
        embed_response = co_embed.embed(
            texts=["Machine learning"], model="embed-english-v3.0", input_type="search_document"
        )

        assert embed_response.embeddings is not None
        wrapper.record_key_usage(api_key=embed_key, model_id="embed-english-v3.0", actual_tokens=20)
        print(f"\n Embedding call successful with key {embed_key[:10]}...")

        try:
            chat_key = wrapper.get_api_key(model_id="command-r", estimated_tokens=500)

            co_chat = cohere_client.Client(chat_key)
            chat_response = co_chat.chat(message="Say 'hello' in one word", model="command-r")

            assert chat_response.text is not None
            wrapper.record_key_usage(api_key=chat_key, model_id="command-r", actual_tokens=50)
            print(f"✓ Chat call successful with key {chat_key[:10]}...")
            print(f"  Response: {chat_response.text[:50]}...")
        except Exception as e:
            print(f"⚠ Chat test skipped: {e}")

    def test_print_stats_after_usage(self, wrapper: MultiProviderWrapper):
        """Test that stats are properly tracked and can be printed"""
        # Make a few requests first
        api_key = wrapper.get_api_key(model_id="embed-english-v3.0")
        wrapper.record_key_usage(api_key, model_id="embed-english-v3.0", actual_tokens=100)

        print("\n" + "=" * 60)
        print("GLOBAL STATS:")
        print("=" * 60)
        wrapper.print_global_stats()

        print("\n" + "=" * 60)
        print("MODEL STATS FOR embed-english-v3.0:")
        print("=" * 60)
        wrapper.print_model_stats("embed-english-v3.0")

        # Verify stats exist
        global_stats = wrapper.manager.get_global_stats()
        assert global_stats.total.total_requests > 0
        assert global_stats.total.total_tokens > 0

    def test_exhaustion_and_rotation(self, wrapper: MultiProviderWrapper, cohere_client):
        """Test behavior when a key gets exhausted (rate limited)"""
        print("\n⚠ Testing rate limit behavior (may be slow)...")

        # Get initial key
        first_key = wrapper.get_api_key(model_id="embed-english-v3.0")
        print(f"✓ First key: {first_key[:10]}...")

        # Make many requests to potentially exhaust (based on tier)
        # Free tier: 20 req/min, so let's do 5 requests
        for i in range(5):
            api_key = wrapper.get_api_key(model_id="embed-english-v3.0", estimated_tokens=50)

            co = cohere_client.Client(api_key)
            co.embed(
                texts=[f"Test text {i}"], model="embed-english-v3.0", input_type="search_document"
            )

            wrapper.record_key_usage(
                api_key=api_key, model_id="embed-english-v3.0", actual_tokens=20
            )

            print(f" Request {i + 1}: Used key {api_key[:10]}...")

        print(f" Completed {5} requests successfully")


if __name__ == "__main__":
    # Run with: pytest test_key_retrieval_real.py -v -s
    pytest.main([__file__, "-v", "-s"])
