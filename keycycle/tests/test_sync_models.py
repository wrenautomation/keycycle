import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("sync_models", Path(__file__).parents[2] / "scripts" / "sync_models.py")
sm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sm)


def test_num():
    assert sm.num("1.2K") == 1200 and sm.num("14.4K") == 14400 and sm.num("1M") == 1_000_000
    assert sm.num("\\-") is None and sm.num("Unlimited") is None and sm.num("`30`") == 30


def test_norm_maps_ai_studio_names_to_ids():
    assert sm.norm("Gemini 3 Flash") == sm.norm("gemini-3-flash-preview")
    assert sm.norm("Gemini 2.5 Flash TTS") == sm.norm("gemini-2.5-flash-preview-tts")
    assert sm.norm("Gemma 4 26B") == sm.norm("gemma-4-26b-a4b-it")
    assert sm.norm("Gemini 3.5 Flash") != sm.norm("gemini-3.5-flash-lite")


def test_limits_derives_hour_and_day():
    assert sm.limits(30, 1000, 8000, tpd=200000) == {
        "requests_per_minute": 30, "requests_per_hour": 1000, "requests_per_day": 1000,
        "tokens_per_minute": 8000, "tokens_per_hour": 200000, "tokens_per_day": 200000}
    assert sm.limits(5)["requests_per_day"] == 7200
