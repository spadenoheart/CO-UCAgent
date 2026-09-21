from scripts.freeze_experiment_baseline import redact_setting


def test_setting_redaction_preserves_non_secret_token_limits():
    source = """openai_api_key: \"sk-example-secret\"
# openai_api_key: \"sk-comment-secret\"
max_context_tokens: 131072
max_tokens: 65536
"""

    redacted = redact_setting(source)

    assert 'openai_api_key: "***REDACTED***"' in redacted
    assert '# openai_api_key: "***REDACTED***"' in redacted
    assert "sk-example-secret" not in redacted
    assert "sk-comment-secret" not in redacted
    assert "max_context_tokens: 131072" in redacted
    assert "max_tokens: 65536" in redacted
