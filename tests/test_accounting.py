"""Unit tests for SQLite token accounting and API key access control."""

from inferops.gateway.accounting import TokenAccountingManager


def test_token_accounting_lifecycle(tmp_path):
    db_file = tmp_path / "test_usage.db"
    mgr = TokenAccountingManager(db_file)

    # 1. Test savings calculation
    savings = mgr.calculate_savings(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert savings == 12.50  # 2.50 + 10.00

    # 2. Record usage without key
    rec1 = mgr.record_usage(
        model="qwen2.5-coder-7b",
        prompt_tokens=1000,
        completion_tokens=500,
        latency_ms=120.5,
    )
    assert rec1.total_tokens == 1500
    assert rec1.estimated_savings_usd > 0

    # 3. Create API key
    raw_key, info = mgr.create_api_key(name="test-service", rate_limit_rpm=120)
    assert raw_key.startswith("sk-inferops-")
    assert info.name == "test-service"
    assert info.rate_limit_rpm == 120

    # 4. Verify API key
    verified = mgr.verify_api_key(raw_key)
    assert verified is not None
    assert verified.key_id == info.key_id

    # Invalid key
    assert mgr.verify_api_key("sk-inferops-invalid-random-bytes") is None

    # 5. Record usage with key
    rec2 = mgr.record_usage(
        model="qwen2.5-coder-7b",
        prompt_tokens=2000,
        completion_tokens=1000,
        latency_ms=250.0,
        api_key_id=info.key_id,
    )
    assert rec2.api_key_id == info.key_id

    # 6. Check summary
    summary = mgr.get_summary_statistics()
    assert summary["total_requests"] == 2
    assert summary["total_prompt_tokens"] == 3000
    assert summary["total_completion_tokens"] == 1500
    assert summary["grand_total_tokens"] == 4500
    assert summary["total_savings_usd"] > 0
    assert len(summary["by_model"]) == 1
    assert summary["by_model"][0]["model"] == "qwen2.5-coder-7b"

    # 7. Revoke API key
    revoked = mgr.revoke_api_key(info.key_id)
    assert revoked is True
    # Now verify should fail
    assert mgr.verify_api_key(raw_key) is None
