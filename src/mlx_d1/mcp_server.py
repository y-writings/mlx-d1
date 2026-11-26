"""MCP tools that forward to a running mlx-d1 HTTP server.

The model stays in the HTTP process. Connect this stdio server from pi with
pi-mcp-adapter; it does not load d1 itself.
"""

from __future__ import annotations

from typing import Any

from mlx_d1.client import DecisionClient, DecisionClientError


def build_mcp(client: DecisionClient) -> Any:
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:
        message = "the mcp package is not installed; install this project to get it"
        raise RuntimeError(message) from exc

    mcp = MCPServer("mlx-d1")

    @mcp.tool()
    def decide(
        questions: dict[str, Any],
        state: str | dict[str, Any] | list[Any] | None = None,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        """Ask named, typed questions about a state and optional images.

        Questions, criteria, and options all come from the caller. Image values
        may be file paths, base64, or data URLs. Paths must sit in the judging
        server's allowed directories. Returns the /v1/systemone JSON body.
        """
        return _call(client.decide, state=state, questions=questions, images=images)

    @mcp.tool()
    def compare(
        a: dict[str, Any],
        b: dict[str, Any],
        criteria: list[Any],
        state: str | dict[str, Any] | list[Any] | None = None,
    ) -> dict[str, Any]:
        """Compare two candidates on caller-supplied criteria.

        Each candidate is an object with optional text and images. criteria is
        a list of instruction strings or objects with name and instructions.
        The server judges both orders and keeps a winner only when the same
        side wins both. A split is a tie. No cutoff is applied.
        """
        return _call(client.compare, a=a, b=b, criteria=criteria, state=state)

    @mcp.tool()
    def run_preset(
        name: str,
        images: list[str] | None = None,
        state: str | dict[str, Any] | list[Any] | None = None,
    ) -> dict[str, Any]:
        """Run a question preset configured on the judging server.

        The preset supplies the questions. Pass the state and images to judge.
        """
        return _call(client.run_preset, name, state=state, images=images)

    return mcp


def _call(func: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return func(*args, **kwargs)
    except DecisionClientError as exc:
        raise RuntimeError(str(exc)) from exc
