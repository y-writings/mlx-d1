import pytest
from fastapi.testclient import TestClient

from mlx_d1.client import DecisionClient, DecisionClientError
from mlx_d1.mcp_server import build_mcp


def test_client_decide_compare_and_preset(client):
    http = DecisionClient("http://test", client=client)
    decided = http.decide(
        state="hello",
        questions={"q": {"type": "noul", "instructions": "Yes?"}},
    )
    assert decided["answers"]["q"]["type"] == "noul"
    compared = http.compare(a={"text": "a"}, b={"text": "b"}, criteria="Which?")
    assert compared["winner"] == "tie"
    with pytest.raises(DecisionClientError, match="unknown preset"):
        http.run_preset("missing", state="x")


def test_mcp_registers_the_three_tools(client):
    mcp = build_mcp(DecisionClient("http://test", client=client))
    tools = mcp._tool_manager.list_tools()
    assert sorted(tool.name for tool in tools) == ["compare", "decide", "run_preset"]


def test_mcp_decide_calls_the_server(client):
    mcp = build_mcp(DecisionClient("http://test", client=client))
    tool = next(tool for tool in mcp._tool_manager.list_tools() if tool.name == "decide")
    result = tool.fn(
        questions={"q": {"type": "noul", "instructions": "Yes?"}},
        state="hello",
        images=None,
    )
    assert result["answers"]["q"]["noul"] == 0.5


def test_client_can_use_a_fresh_test_client(app):
    with TestClient(app) as http:
        body = DecisionClient("http://test", client=http).decide(
            state="ACME SA",
            questions={"q": {"type": "text", "instructions": "What is written?"}},
        )
    assert body["answers"]["q"]["type"] == "text"
