"""Regression coverage for safe upstream error classification."""

import json

from doubao2api.upstream_errors import parse_upstream_error


def test_classifies_explicit_rate_limit_from_status_and_payload():
    error = parse_upstream_error(
        {"error_code": 429, "error_type": "rate_limit", "log_id": "abc-123"}
    )

    assert error.category == "rate_limit"
    assert error.status_code == 429
    assert error.error_type == "rate_limit"
    assert error.log_id == "abc-123"


def test_classifies_real_verify_decision_string_ahead_of_generic_rate_wording():
    secret = "challenge=secret-token; cookie=session; signature=private"
    error = parse_upstream_error({
        "error_code": 710022004,
        "error_msg": f"rate limited {secret}",
        "extra": {
            "decision": json.dumps({
                "type": "verify",
                "subtype": "semantic_reasoning",
                "log_id": "log-456",
            })
        },
    })

    assert error.category == "verification"
    assert error.status_code == 503
    assert error.error_type == "verify"
    assert error.subtype == "semantic_reasoning"
    rendered = str(error.to_openai_error("http://127.0.0.1:9090/admin"))
    assert "log-456" in rendered
    assert "710022004" in rendered
    assert all(term not in rendered for term in ("secret-token", "session", "private"))


def test_classifies_dict_decision_and_handles_malformed_decision():
    verify = parse_upstream_error({
        "error_msg": "rate limited",
        "extra": {"decision": {"type": "verify", "subtype": "semantic_reasoning"}},
    })
    malformed = parse_upstream_error({
        "error_msg": "rate limited",
        "extra": {"decision": "not-json"},
    })

    assert (verify.category, verify.status_code) == ("verification", 503)
    assert (malformed.category, malformed.status_code) == ("rate_limit", 429)

    explicit_rate_limit = parse_upstream_error(
        {"extra": {"decision": {"type": "verify"}}}, status_code=429
    )
    assert (explicit_rate_limit.category, explicit_rate_limit.status_code) == ("rate_limit", 429)


def test_classifies_expired_session_and_unknown_or_malformed_payloads():
    expired = parse_upstream_error('{"message":"session expired"}')
    unknown = parse_upstream_error("not-json")
    missing = parse_upstream_error(None)

    assert (expired.category, expired.status_code) == ("session_expired", 401)
    assert (unknown.category, unknown.status_code) == ("unknown", 502)
    assert (missing.category, missing.status_code) == ("unknown", 502)


def test_710022004_alone_is_not_treated_as_verification():
    error = parse_upstream_error({"error_code": 710022004})

    assert (error.category, error.status_code) == ("unknown", 502)
