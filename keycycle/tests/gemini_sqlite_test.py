import asyncio
import os
from pathlib import Path

import pytest
from agno.agent import Agent
from dotenv import load_dotenv

from keycycle import MultiProviderWrapper

# Add project root to sys.path


load_dotenv(dotenv_path="./local.env", override=True)


@pytest.mark.anyio
async def test_gemini_sqlite():
    print("--- Starting Gemini SQLite Test ---")

    # 1. Setup SQLite DB Path
    db_path = Path(__file__).parent / "test_usage_gemini.db"
    db_url = f"sqlite:///{db_path}"

    # Ensure clean state
    if db_path.exists():
        try:
            os.remove(db_path)
            print(f"Removed existing DB: {db_path}")
        except Exception as e:
            print(f"Warning: Could not remove existing DB: {e}")

    print(f"Using DB URL: {db_url}")

    # 2. Initialize Wrapper with SQLite URL
    try:
        gemini_wrapper = MultiProviderWrapper.from_env(
            provider="gemini",
            default_model_id="gemini-2.5-flash",
            env_file="./local.env",
            db_url=db_url,  # Passing the SQLite URL here
        )
        print("Wrapper initialized successfully.")
    except Exception as e:
        print(f"FAILED to initialize wrapper: {e}")
        return

    try:
        # 3. Get Model & Make Request
        print("Getting model...")
        model = gemini_wrapper.get_model(estimated_tokens=100)

        agent = Agent(model=model, markdown=True)

        prompt = "Hello! Please confirm you are working with a short sentence."
        print(f"Sending prompt: '{prompt}'")

        # Using aprint_response to trigger the async flow
        response = await agent.aprint_response(prompt, stream=False)
        print("\nRequest completed.")

        # 4. Verify DB logging
        # We need to wait a moment for the async logger to flush to the SQLite file
        print("Waiting for async logger to flush...")
        await asyncio.sleep(2)

        # Stop wrapper to ensure flush
        gemini_wrapper.manager.stop()

        # Check if DB file exists
        if db_path.exists():
            print(f"SUCCESS: DB file created at {db_path}")

            # Verify content
            from keycycle.usage.db_logic import UsageDatabase
            # We can use the db instance from the wrapper or create a new one to read
            # The wrapper's db engine might be disposed by stop(), so let's verify file size or reconnect

            # Simple check: File size > 0
            size = db_path.stat().st_size
            print(f"DB File Size: {size} bytes")

            if size > 0:
                print("DB file is not empty.")
            else:
                print("FAILURE: DB file is empty.")

            # Optional: Read back records
            # Re-connect to read
            read_db = UsageDatabase(db_url=db_url)
            # Fetch all history for provider 'gemini'
            history = read_db.load_provider_history("gemini", seconds_lookback=60)
            print(f"Retrieved {len(history)} records from DB.")
            for record in history:
                print(f" - {record}")

            if len(history) > 0:
                print("SUCCESS: Usage logged and retrieved.")
            else:
                print("FAILURE: No usage records found in DB.")

        else:
            print("FAILURE: DB file was not created.")

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        # Close any lingering connections
        if "gemini_wrapper" in locals():
            try:
                # Ensure manager is stopped if we didn't reach that line
                gemini_wrapper.manager.stop()
            except Exception:
                pass

        if "read_db" in locals():
            read_db.engine.dispose()


if __name__ == "__main__":
    asyncio.run(test_gemini_sqlite())
