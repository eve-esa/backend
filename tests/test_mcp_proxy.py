from src.routers.mcp_proxy import _is_error_message, _tool_call_outcome


def test_is_error_message_json_rpc_error():
    assert _is_error_message([{"error": {"code": -32603, "message": "boom"}}]) is True


def test_is_error_message_result_is_error():
    assert _is_error_message([{"result": {"isError": True, "content": []}}]) is True


def test_is_error_message_success():
    assert _is_error_message([{"result": {"isError": False, "content": []}}]) is False


def test_is_error_message_unparseable_returns_none():
    assert _is_error_message(None) is None


def test_tool_call_outcome_unknown_when_mcp_unparseable_and_http_ok():
    assert _tool_call_outcome(http_failed=False, is_error=None) == "unknown"


def test_tool_call_outcome_error_when_http_failed_even_if_mcp_unknown():
    assert _tool_call_outcome(http_failed=True, is_error=None) == "error"


def test_tool_call_outcome_success_only_when_mcp_confirmed_ok():
    assert _tool_call_outcome(http_failed=False, is_error=False) == "success"
