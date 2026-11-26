"""HTTP client used by the MCP server and the latency command."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx


class DecisionClientError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class DecisionClient:
    """Call a running mlx-d1 server. This client does not load the model."""

    def __init__(self, base_url: str, timeout: float = 120.0, client: Any | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = (
            client if client is not None else httpx.Client(base_url=self.base_url, timeout=timeout)
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def decide(
        self,
        *,
        state: Any = None,
        questions: dict[str, Any],
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"state": state, "questions": questions}
        if images is not None:
            payload["images"] = images
        return self._post("/v1/systemone", payload)

    def compare(
        self,
        *,
        a: dict[str, Any],
        b: dict[str, Any],
        criteria: Any,
        state: Any = None,
    ) -> dict[str, Any]:
        return self._post("/v1/compare", {"a": a, "b": b, "criteria": criteria, "state": state})

    def run_preset(
        self,
        name: str,
        *,
        state: Any = None,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"state": state}
        if images is not None:
            payload["images"] = images
        return self._post(f"/v1/presets/{quote(name, safe='')}", payload)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(path, json=payload)
        except httpx.HTTPError as exc:
            raise DecisionClientError(
                0, f"could not reach the judging server at {self.base_url}: {exc}"
            ) from exc
        if response.status_code >= 400:
            raise DecisionClientError(response.status_code, error_message(response))
        try:
            body = response.json()
        except ValueError as exc:
            raise DecisionClientError(
                response.status_code, "judging server returned non-JSON"
            ) from exc
        if not isinstance(body, dict):
            raise DecisionClientError(
                response.status_code, "judging server returned a non-object JSON body"
            )
        return body


def error_message(response: Any) -> str:
    try:
        body = response.json()
    except ValueError:
        text = getattr(response, "text", "")
        return text.strip() or f"HTTP {response.status_code}"
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if body.get("detail"):
            return str(body["detail"])
    return f"HTTP {response.status_code}"
