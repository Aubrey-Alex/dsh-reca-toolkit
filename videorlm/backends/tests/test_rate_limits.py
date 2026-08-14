"""Tests for ``_common.dashscope_sdk`` rate-limit loader.

Verifies the 3-layer priority:

    hardcoded fallback  <  configs/dashscope_rate_limits.json  <  SUBMIT_* env vars

The bundled JSON at ``unirlm-02/configs/dashscope_rate_limits.json`` carries
the official Aliyun rate-limit numbers (2026-05-16 fetch) + happyhorse
empirical numbers. ``reload_model_limits()`` is the seam tests use to
re-resolve the table after monkey-patching env vars.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from videorlm.backends._common import dashscope_sdk as ds


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Clear the JSON-path env override so each test starts from the
    bundled JSON. Per-value env overrides no longer exist — to customize
    rate-limit values, edit the JSON or swap the file via
    ``DASHSCOPE_RATE_LIMITS_CONFIG=<path>``."""
    monkeypatch.delenv("DASHSCOPE_RATE_LIMITS_CONFIG", raising=False)
    yield
    ds.reload_model_limits()


def _reload():
    return ds.reload_model_limits()


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: bundled JSON loads with official Aliyun numbers
# ─────────────────────────────────────────────────────────────────────────────


def test_bundled_json_path_exists():
    """The repo ships with the official rate-limit table — accessible via
    the dashscope provider's ``rate_limits_path`` field."""
    from videorlm.backends._common.providers import get_provider
    path = get_provider("dashscope").rate_limits_path
    assert path and path.exists(), f"bundled JSON missing at {path}"
    # And it's valid JSON
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "models" in data
    assert isinstance(data["models"], dict)


def test_wan27_video_models_match_official_5rps_5concur():
    """Aliyun official: wan2.7-{t2v,i2v,r2v,image,image-pro} all
    5 RPS / 5 concurrent."""
    table = _reload()
    for model in (
        "wan2.7-t2v", "wan2.7-i2v", "wan2.7-r2v",
        "wan2.7-image", "wan2.7-image-pro",
    ):
        assert model in table, f"{model} missing from JSON"
        assert table[model]["interval"] == 0.2, \
            f"{model} interval != 0.2s (= 1/5RPS): got {table[model]}"
        assert table[model]["max_parallel"] == 5, \
            f"{model} max_parallel != 5: got {table[model]}"


def test_wan22_models_match_official_2rps_2concur():
    table = _reload()
    for model in (
        "wan2.2-t2v-plus", "wan2.2-i2v-plus", "wan2.2-i2v-flash",
        "wan2.2-kf2v-flash", "wan2.2-t2i-plus", "wan2.2-t2i-flash",
    ):
        assert table[model]["interval"] == 0.5, model
        assert table[model]["max_parallel"] == 2, model


def test_wanx_v1_official_2rps_1concur():
    """wanx-v1: 2 RPS / 1 concurrent — single in-flight slot."""
    table = _reload()
    assert table["wanx-v1"]["interval"] == 0.5
    assert table["wanx-v1"]["max_parallel"] == 1


def test_qwen_image_2_0_pro_official_2_per_minute():
    """qwen-image-2.0-pro: 2 次/分钟 = 30s interval. The unit caveat
    documented in JSON _meta — confirm the math is right in code."""
    table = _reload()
    assert table["qwen-image-2.0-pro"]["interval"] == 30.0
    # max_parallel placeholder for 同步接口无限制
    assert table["qwen-image-2.0-pro"]["max_parallel"] == 100


def test_qwen_image_2_0_official_2_per_sec():
    """Non-pro variant is 2/s = 0.5s interval."""
    table = _reload()
    assert table["qwen-image-2.0"]["interval"] == 0.5


def test_happyhorse_uses_empirical_values():
    """happyhorse-1.0-r2v: 16 slots per MEMORY (not in Aliyun official table)."""
    table = _reload()
    assert table["happyhorse-1.0-r2v"]["max_parallel"] == 16
    assert table["happyhorse-1.0-i2v"]["max_parallel"] == 8
    assert table["happyhorse-1.0-t2v"]["max_parallel"] == 8
    assert table["happyhorse-1.0-video-edit"]["max_parallel"] == 8


def test_kind_aliases_present():
    """Legacy shortcut keys t2v / i2v / t2i / i2i / image still resolve so
    callers passing ``rate_key="t2v"`` (not a specific model name) still work."""
    table = _reload()
    for kind in ("t2v", "i2v", "t2i", "i2i", "image"):
        assert kind in table, f"alias {kind!r} missing"


# ─────────────────────────────────────────────────────────────────────────────
# JSON file path env (DASHSCOPE_RATE_LIMITS_CONFIG) — file selector, not
# per-value override. Value overrides via env are intentionally NOT
# supported — edit the JSON to change rate-limit values.
# ─────────────────────────────────────────────────────────────────────────────


def test_env_override_nonexistent_falls_back_to_bundled(monkeypatch, tmp_path):
    """``DASHSCOPE_RATE_LIMITS_CONFIG`` pointing at a nonexistent file
    should TRANSPARENTLY fall through to the bundled
    ``configs/dashscope_rate_limits.json`` — broken env override never
    drops table entries silently."""
    fake = tmp_path / "does-not-exist.json"
    monkeypatch.setenv("DASHSCOPE_RATE_LIMITS_CONFIG", str(fake))
    table = _reload()
    # Bundled JSON still loaded (env failure was transparent).
    assert "wan2.7-i2v" in table
    assert table["wan2.7-i2v"]["interval"] == 0.2
    # Hardcoded kind aliases still present.
    for kind in ("t2v", "i2v", "t2i", "i2i", "image"):
        assert kind in table


def test_malformed_env_json_falls_back_to_bundled(monkeypatch, tmp_path):
    """A broken env-pointed JSON returns empty → ``rate_limits()`` then
    falls through to the bundled JSON. Never leaves the caller with
    an empty table."""
    bad = tmp_path / "broken.json"
    bad.write_text("this is { not valid json", encoding="utf-8")
    monkeypatch.setenv("DASHSCOPE_RATE_LIMITS_CONFIG", str(bad))
    table = _reload()
    # Bundled JSON entries still loaded.
    assert "t2v" in table
    assert "wan2.7-i2v" in table


def test_custom_json_path_loads(monkeypatch, tmp_path):
    """A user can supply their own JSON file and it overlays the bundled one."""
    custom = tmp_path / "custom.json"
    custom.write_text(json.dumps({
        "models": {
            "wan2.7-i2v": {"interval_s": 3.14, "max_parallel": 42},
        }
    }), encoding="utf-8")
    monkeypatch.setenv("DASHSCOPE_RATE_LIMITS_CONFIG", str(custom))
    table = _reload()
    assert table["wan2.7-i2v"]["interval"] == 3.14
    assert table["wan2.7-i2v"]["max_parallel"] == 42


def test_individual_malformed_row_skipped_others_kept(monkeypatch, tmp_path):
    """One bad row shouldn't kill the rest of the table."""
    custom = tmp_path / "mixed.json"
    custom.write_text(json.dumps({
        "models": {
            "wan2.7-i2v": {"interval_s": "garbage"},        # bad
            "wan2.7-t2v": {"interval_s": 0.5, "max_parallel": 9},  # good
            "wan2.7-r2v": {"max_parallel": 5},               # missing interval
        }
    }), encoding="utf-8")
    monkeypatch.setenv("DASHSCOPE_RATE_LIMITS_CONFIG", str(custom))
    table = _reload()
    assert table.get("wan2.7-t2v") == {"interval": 0.5, "max_parallel": 9}
    # Bad rows: NOT loaded from custom JSON; not in hardcoded either.
    assert "wan2.7-i2v" not in table
    assert "wan2.7-r2v" not in table


# ─────────────────────────────────────────────────────────────────────────────
# RateLimiter integration
# ─────────────────────────────────────────────────────────────────────────────


def test_rate_limiter_consumes_loaded_table_for_wan27_i2v():
    """The dashscope-bound ``cached_rate_limiter`` should resolve model
    limits from the provider's JSON. Bare ``RateLimiter()`` is
    provider-agnostic and falls back to defaults — that's the
    ``test_rate_limiter_unknown_model_uses_global_default`` case."""
    from videorlm.backends._common.rate_limiter import cached_rate_limiter
    _reload()
    rl = cached_rate_limiter("dashscope")
    interval, max_parallel = rl._resolve_limits("wan2.7-i2v")
    assert interval == 0.2
    assert max_parallel == 5


def test_rate_limiter_unknown_model_uses_global_default():
    rl = ds.RateLimiter()
    interval, max_parallel = rl._resolve_limits("model-that-does-not-exist")
    # Falls back to rl.default_interval / rl.default_parallel
    assert interval == ds.SUBMIT_MIN_INTERVAL_S
    assert max_parallel == ds.SUBMIT_MAX_PARALLEL_DEFAULT
