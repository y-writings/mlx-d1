"""The decision backend interface.

``system_one`` answers one state. ``system_one_batch`` answers several states
in one model call. Implementations must not run those calls concurrently with
themselves; the HTTP service also holds a gate around every call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class DecisionBackend(Protocol):
    model_id: str
    name: str

    def system_one(
        self,
        state: Any,
        questions: Mapping[str, Any],
        media: Sequence[Any] | None = None,
        *,
        score_readout: str | None = None,
        set_size: int | None = None,
    ) -> dict[str, Any]:
        """Return ``{"answers": {...}, "usage": {...}}``."""

    def system_one_batch(
        self,
        requests: Sequence[tuple[Any, Mapping[str, Any], Sequence[Any] | None]],
        *,
        score_readout: str | None = None,
        set_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return one result per request, in order."""

    def close(self) -> None:
        """Release weights or other resources. May be a no-op."""
