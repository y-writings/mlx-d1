"""An in-process backend that returns shaped answers and records calls.

Use it for tests and for ``mlx-d1 serve --backend fake``. It does not load
weights and it does not imitate model quality. A custom ``responder`` can
script compare behavior.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

Responder = Callable[[Any, Mapping[str, Any], Sequence[Any] | None], dict[str, Any]]


class FakeBackend:
    name = "fake"

    def __init__(
        self,
        model_id: str = "fake-d1",
        *,
        responder: Responder | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self.model_id = model_id
        self.delay_s = delay_s
        self.responder = responder or default_responder
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._in_flight = 0
        self.max_in_flight = 0

    def system_one(
        self,
        state: Any,
        questions: Mapping[str, Any],
        media: Sequence[Any] | None = None,
        *,
        score_readout: str | None = None,
        set_size: int | None = None,
    ) -> dict[str, Any]:
        self._enter()
        try:
            if self.delay_s:
                time.sleep(self.delay_s)
            call = {
                "kind": "one",
                "state": state,
                "questions": dict(questions),
                "media": None if not media else list(media),
                "score_readout": score_readout,
                "set_size": set_size,
            }
            with self._lock:
                self.calls.append(call)
            return self.responder(state, questions, media)
        finally:
            self._leave()

    def system_one_batch(
        self,
        requests: Sequence[tuple[Any, Mapping[str, Any], Sequence[Any] | None]],
        *,
        score_readout: str | None = None,
        set_size: int | None = None,
    ) -> list[dict[str, Any]]:
        self._enter()
        try:
            if self.delay_s:
                time.sleep(self.delay_s)
            recorded = []
            results = []
            for state, questions, media in requests:
                recorded.append(
                    {
                        "state": state,
                        "questions": dict(questions),
                        "media": None if not media else list(media),
                    }
                )
                results.append(self.responder(state, questions, media))
            with self._lock:
                self.calls.append(
                    {
                        "kind": "batch",
                        "requests": recorded,
                        "score_readout": score_readout,
                        "set_size": set_size,
                    }
                )
            return results
        finally:
            self._leave()

    def close(self) -> None:
        return None

    def _enter(self) -> None:
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)

    def _leave(self) -> None:
        with self._lock:
            self._in_flight -= 1


def default_responder(
    state: Any,
    questions: Mapping[str, Any],
    media: Sequence[Any] | None,
) -> dict[str, Any]:
    """Return a typed answer for every question. The values are placeholders."""
    del state, media
    return {
        "answers": {name: canned_answer(question) for name, question in questions.items()},
        "usage": {"input_tokens": 1, "output_tokens": 0},
    }


def canned_answer(question: Mapping[str, Any]) -> dict[str, Any]:
    qtype = question.get("type")
    if qtype == "noul":
        return {"type": "noul", "noul": 0.5}
    if qtype == "choice" and question.get("multiple") is True:
        criteria = question.get("criteria") or {}
        names = list(criteria) if isinstance(criteria, dict) else []
        return {"type": "choice", "choices": [], "probabilities": {name: 0.0 for name in names}}
    if qtype == "choice":
        criteria = question.get("criteria") or {}
        names = list(criteria) if isinstance(criteria, dict) else []
        choice = names[0] if names else "FIRST"
        if len(names) <= 1:
            probabilities = {choice: 1.0}
        else:
            rest = 0.2 / (len(names) - 1)
            probabilities = {name: (0.8 if name == choice else rest) for name in names}
        return {
            "type": "choice",
            "choice": choice,
            "probabilities": probabilities,
            "confidence": 0.6,
        }
    if qtype == "score":
        levels = question.get("criteria") or []
        if not isinstance(levels, list):
            levels = []
        legend = {str(index): level for index, level in enumerate(levels)}
        probabilities = {str(index): (1.0 if index == 0 else 0.0) for index in range(len(levels))}
        return {
            "type": "score",
            "score": 0,
            "confidence": 1.0,
            "probabilities": probabilities,
            "legend": legend,
        }
    if qtype == "text":
        return {"type": "text", "text": None, "span": None, "found": 0.0}
    if qtype == "date":
        return {"type": "date", "date": None, "text": None, "span": None, "found": 0.0}
    if qtype == "number" and "range" in question:
        return {"type": "number", "number": 0, "interval": [0, 0], "unit": question.get("unit")}
    if qtype == "number":
        return {
            "type": "number",
            "number": None,
            "unit": None,
            "text": None,
            "span": None,
            "found": 0.0,
        }
    if qtype == "box":
        return {"type": "box", "box": None, "found": 0.0, "confidence": 0.0}
    if qtype == "boxes":
        return {"type": "boxes", "boxes": [], "count": 0, "at_least": False, "presence": []}
    return {"type": qtype}


def choice_answer(choice: str, *, confidence: float = 0.8) -> dict[str, Any]:
    """A FIRST/SECOND choice answer for compare tests."""
    other = "SECOND" if choice.upper() == "FIRST" else "FIRST"
    selected = "FIRST" if choice.upper() == "FIRST" else "SECOND"
    return {
        "type": "choice",
        "choice": selected,
        "probabilities": {selected: 0.8, other: 0.2},
        "confidence": confidence,
    }
