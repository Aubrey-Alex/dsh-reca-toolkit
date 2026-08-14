"""Tests for ``backends._common.providers`` — the provider layer.

Covers:
  - 5 built-in providers auto-register (dashscope / openai / kling /
    pixverse / vidu).
  - ``get_provider`` is case-insensitive and rejects unknown names.
  - ``register_provider`` rejects duplicates unless ``replace=True``.
  - ``Provider.api_key()`` walks ``api_key_envs`` in order.
  - ``Provider.rate_limits()`` loads the bundled JSON.
  - ``rate_limits_env_override`` (e.g. ``DASHSCOPE_RATE_LIMITS_CONFIG``)
    trumps the bundled path when the env points at a real file.
  - ``with_key`` auto-pulls the provider's ``response_classifier`` when
    the caller omits ``classify_response`` (the key delivery property).
  - Backend wiring: ``GPTImage2Backend`` reports
    ``caps.provider == "openai"``; ``DashScopeQwenAgent`` /
    ``OpenAICompatibleAgent`` carry the right ``PROVIDER_NAME``.
"""
from __future__ import annotations

import json
import os

import pytest

from videorlm.backends._common import key_pool as _key_pool_mod
from videorlm.backends._common.providers import (
    Provider,
    get_provider,
    list_providers,
    register_provider,
)
from videorlm.backends._common.platforms import (
    get_platform,
    PlatformProfile,
    with_key,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_pools_and_env(monkeypatch):
    """Reset the cached KeyPool registry + clear env-override envs each test."""
    monkeypatch.setattr(_key_pool_mod, "_POOLS", {})
    monkeypatch.setattr(_key_pool_mod, "_HEALTH_THREAD_STARTED", False)
    monkeypatch.setenv("RECA_KEYPOOL_HEALTH_DUMP", "0")
    for env in (
        "DASHSCOPE_RATE_LIMITS_CONFIG",
        "OPENAI_RATE_LIMITS_CONFIG",
    ):
        monkeypatch.delenv(env, raising=False)
    yield


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


def test_five_builtin_providers_registered():
    """The package auto-imports dashscope / openai / _passthrough at
    import time — the registry should always carry these 5."""
    names = set(list_providers())
    assert names == {"dashscope", "openai", "kling", "pixverse", "vidu"}, names


def test_get_provider_case_insensitive_and_strips_whitespace():
    assert get_provider("DashScope").name == "dashscope"
    assert get_provider("  openai  ").name == "openai"
    assert get_provider("KLING").name == "kling"


def test_get_provider_unknown_raises_with_available_list():
    with pytest.raises(KeyError, match="unknown provider 'no-such'"):
        get_provider("no-such")
    # Error message should list the registered names so devs find the typo.
    try:
        get_provider("no-such")
    except KeyError as exc:
        assert "dashscope" in str(exc)
        assert "openai" in str(exc)


def test_register_provider_rejects_duplicate_without_replace():
    custom = Provider(name="dashscope", api_key_envs=("X",))
    with pytest.raises(ValueError, match="already registered"):
        register_provider(custom)


def test_register_provider_replace_true_overwrites():
    """``replace=True`` is the documented escape hatch for tests / dev."""
    original = get_provider("kling")
    try:
        sub = Provider(name="kling", api_key_envs=("ONLY_FOR_TEST",))
        register_provider(sub, replace=True)
        assert get_provider("kling") is sub
    finally:
        # Restore so we don't pollute other tests.
        register_provider(original, replace=True)


# ─────────────────────────────────────────────────────────────────────────────
# Back-compat aliases
# ─────────────────────────────────────────────────────────────────────────────


def test_platform_profile_is_provider_alias():
    """``PlatformProfile`` was the pre-refactor name; new code uses
    ``Provider`` but ``PlatformProfile`` must keep working — 30+ legacy
    imports rely on it."""
    assert PlatformProfile is Provider


def test_get_platform_is_get_provider_alias():
    assert get_platform is get_provider


def test_provider_dot_provider_attr_aliases_name():
    """``_PLATFORM.provider`` is read in 11+ backend files to set
    ``BackendCapabilities.provider=...`` — must still work."""
    ds = get_provider("dashscope")
    assert ds.provider == "dashscope"
    assert ds.provider == ds.name


# ─────────────────────────────────────────────────────────────────────────────
# Key resolution + multi-key pool
# ─────────────────────────────────────────────────────────────────────────────


def test_api_key_reads_openai_api_key(monkeypatch):
    """The single env name in ``api_key_envs`` resolves to ``api_key()``."""
    monkeypatch.delenv("OPENAI_API_KEYS", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-single")
    op = get_provider("openai")
    assert op.api_key() == "sk-single"


def test_openai_pool_fans_out_via_csv(monkeypatch):
    """Multi-key path is the ``OPENAI_API_KEYS=k1,k2,...`` CSV."""
    monkeypatch.setattr(_key_pool_mod, "_POOLS", {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEYS", "sk-a,sk-b")
    pool = get_provider("openai").key_pool()
    keys = {s.key for s in pool.keys}
    assert keys == {"sk-a", "sk-b"}, keys


# ─────────────────────────────────────────────────────────────────────────────
# rate_limits JSON loading + env override
# ─────────────────────────────────────────────────────────────────────────────


def test_dashscope_rate_limits_loads_bundled_json():
    table = get_provider("dashscope").rate_limits()
    assert "wan2.7-i2v" in table
    assert table["wan2.7-i2v"]["interval"] == 0.2
    assert table["wan2.7-i2v"]["max_parallel"] == 5


def test_openai_rate_limits_loads_bundled_json():
    table = get_provider("openai").rate_limits()
    assert "gpt-image-2" in table
    assert table["gpt-image-2"]["interval"] == 2.0
    assert table["gpt-image-2"]["max_parallel"] == 8


def test_env_override_beats_bundled_json(monkeypatch, tmp_path):
    """``DASHSCOPE_RATE_LIMITS_CONFIG=<file>`` swaps the whole table."""
    custom = tmp_path / "my-limits.json"
    custom.write_text(json.dumps({
        "models": {"wan2.7-i2v": {"interval_s": 99.9, "max_parallel": 1}}
    }), encoding="utf-8")
    monkeypatch.setenv("DASHSCOPE_RATE_LIMITS_CONFIG", str(custom))
    table = get_provider("dashscope").rate_limits()
    assert table["wan2.7-i2v"] == {"interval": 99.9, "max_parallel": 1}


def test_env_override_nonexistent_falls_back_to_bundled(monkeypatch, tmp_path):
    """Env pointing at a missing file shouldn't drop entries — fall
    transparently to the bundled JSON."""
    fake = tmp_path / "missing.json"
    monkeypatch.setenv("DASHSCOPE_RATE_LIMITS_CONFIG", str(fake))
    table = get_provider("dashscope").rate_limits()
    assert "wan2.7-i2v" in table   # bundled JSON still loaded


# ─────────────────────────────────────────────────────────────────────────────
# with_key auto-pulls provider classifier
# ─────────────────────────────────────────────────────────────────────────────


def test_with_key_auto_uses_provider_classifier_dashscope(monkeypatch):
    """When the caller omits ``classify_response``, ``with_key`` should
    auto-use the provider's own classifier. For dashscope, a 429 throttle
    response → ``rate_limit`` kind → cooldown set on the key."""
    monkeypatch.setattr(_key_pool_mod, "_POOLS", {})
    monkeypatch.setenv("DASHSCOPE_API_KEYS", "sk-x,sk-y")

    class FakeThrottle:
        status_code = 429
        code = "Throttling.RateQuotaExceeded"
        message = "slow down"

    with_key("dashscope", lambda k: FakeThrottle())   # no classify_response!
    pool = get_provider("dashscope").key_pool()
    cooling = [s for s in pool.stats() if s["cooldown_remaining_s"] > 0]
    assert len(cooling) == 1, "auto-classifier didn't apply cooldown"


def test_with_key_auto_uses_provider_classifier_openai(monkeypatch):
    """OpenAI provider's classifier distinguishes 429-rate vs 429-quota."""
    monkeypatch.setattr(_key_pool_mod, "_POOLS", {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEYS", "sk-o1,sk-o2")

    class QuotaResp:
        status_code = 429
        text = "you exceeded your current quota"

    with_key("openai", lambda k: QuotaResp())   # no classify_response!
    pool = get_provider("openai").key_pool()
    # 'daily_quota' is permanent cooldown — one key should be cooled inf.
    import math
    cooling = [s for s in pool.stats() if math.isinf(s["cooldown_remaining_s"])]
    assert len(cooling) == 1, "openai classifier didn't route 429-quota to daily_quota"


def test_explicit_classify_response_overrides_provider_default(monkeypatch):
    """Caller-supplied ``classify_response`` MUST win over the provider's
    auto-wired classifier (escape hatch for Retry-After tuple form)."""
    monkeypatch.setattr(_key_pool_mod, "_POOLS", {})
    monkeypatch.setenv("DASHSCOPE_API_KEYS", "sk-z")
    custom_calls: list = []

    def _custom(rsp):
        custom_calls.append(rsp)
        return "ok"

    with_key("dashscope", lambda k: "ignored", classify_response=_custom)
    assert custom_calls == ["ignored"], "explicit classifier was bypassed"


# ─────────────────────────────────────────────────────────────────────────────
# Wiring sanity: agents + media backend pick the right provider
# ─────────────────────────────────────────────────────────────────────────────


def test_openai_compatible_agent_provider_name_is_openai():
    from videorlm.backends.llm.agents.openai_compat.agent import OpenAICompatibleAgent
    assert OpenAICompatibleAgent.PROVIDER_NAME == "openai"


def test_dashscope_qwen_agent_provider_name_is_dashscope():
    from videorlm.backends.llm.agents.qwen.agent import DashScopeQwenAgent
    from videorlm.backends.llm.agents.openai_compat.agent import OpenAICompatibleAgent
    assert DashScopeQwenAgent.PROVIDER_NAME == "dashscope"
    # No per-subclass ``_chat_create`` override — inherits from base.
    assert DashScopeQwenAgent._chat_create is OpenAICompatibleAgent._chat_create


# ─────────────────────────────────────────────────────────────────────────────
# with_key(model=) per-model rate limiter integration
# ─────────────────────────────────────────────────────────────────────────────


def test_with_key_model_arg_enforces_provider_rate_limit(monkeypatch):
    """``with_key("openai", call, model="gpt-image-2")`` should serialize
    5 calls to ≥ 4 × 2.0 s = 8 s based on the JSON's ``interval_s=2.0``."""
    import time
    monkeypatch.setattr(_key_pool_mod, "_POOLS", {})
    from videorlm.backends._common.rate_limiter import reset_rate_limiters_for_tests
    reset_rate_limiters_for_tests()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    t0 = time.time()
    for _ in range(5):
        with_key("openai", lambda k: None, model="gpt-image-2")
    elapsed = time.time() - t0
    # 5 sequential calls × 2.0 s interval (4 gaps × 2.0) → ≥ 8 s
    assert elapsed >= 7.5, f"rate limit not applied; finished in {elapsed:.2f}s"


def test_with_key_no_model_skips_rate_limiter(monkeypatch):
    """``with_key`` without ``model=`` must NOT apply rate gating —
    back-compat for callers that didn't opt in."""
    import time
    monkeypatch.setattr(_key_pool_mod, "_POOLS", {})
    from videorlm.backends._common.rate_limiter import reset_rate_limiters_for_tests
    reset_rate_limiters_for_tests()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    t0 = time.time()
    for _ in range(5):
        with_key("openai", lambda k: None)   # no model=
    elapsed = time.time() - t0
    assert elapsed < 1.0, f"unexpected rate gating w/o model=: {elapsed:.2f}s"


def test_with_key_model_releases_rate_gate_on_exception(monkeypatch):
    """Even if call raises, the per-model rate slot must be released."""
    monkeypatch.setattr(_key_pool_mod, "_POOLS", {})
    from videorlm.backends._common.rate_limiter import (
        cached_rate_limiter,
        reset_rate_limiters_for_tests,
    )
    reset_rate_limiters_for_tests()
    monkeypatch.setenv("DASHSCOPE_API_KEYS", "sk-d")

    def _boom(k):
        raise RuntimeError("kaboom")

    for _ in range(3):
        with pytest.raises(RuntimeError, match="kaboom"):
            with_key("dashscope", _boom, model="wan2.7-i2v")

    # The semaphore should be back to full capacity (max_parallel=5),
    # otherwise the 6th call would hang.
    rl = cached_rate_limiter("dashscope")
    state = rl._get_state("wan2.7-i2v")
    # All 5 permits acquirable → no leak
    permits = sum(1 for _ in range(5) if state.sem.acquire(blocking=False))
    assert permits == 5, f"rate gate leaked: only {permits}/5 permits available"
    for _ in range(permits):
        state.sem.release()


def test_gpt_image_2_backend_reports_openai_provider():
    """``GPTImage2Backend`` was previously wired to a ``gateway`` provider;
    after the rename, ``caps.provider`` must be ``"openai"`` (the
    generic OpenAI-compatible provider, routable via OPENAI_BASE_URL)."""
    from videorlm.backends.media.impl.openai.image.gpt_image_2 import GPTImage2Backend
    caps = GPTImage2Backend().capabilities()
    assert caps.provider == "openai", \
        f"expected provider='openai', got {caps.provider!r}"
