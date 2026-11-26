"""MLX-VLM backend for LiquidAI d1.

d1 does not generate tokens. ``mlx_vlm.generate`` and the chat server cannot
run it. Each decision calls ``model.system_one`` or ``model.system_one_batch``.
A thread lock makes a second call wait even if the HTTP gate is bypassed.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mlx_d1.weights import offline_hub, require_checkpoint


class MlxVlmBackend:
    name = "mlx"

    def __init__(self, model_id: str, *, weights_path: Path) -> None:
        try:
            from mlx_vlm import load
        except ImportError as exc:
            raise RuntimeError(
                'mlx-vlm is not installed. On Apple Silicon run: pip install "mlx-vlm>=0.7.11"'
            ) from exc
        # ``load`` downloads a repo id. Only an existing checkpoint directory is passed,
        # and the Hub is forced offline for the call so a missing file cannot be fetched.
        local = require_checkpoint(Path(weights_path))
        self.model_id = model_id
        self.weights_path = local
        self._infer_lock = threading.Lock()
        # ``load`` attaches the processor and calibration to the model. Keep
        # the processor referenced for the life of the model.
        with offline_hub():
            self._model, self._processor = load(str(local))

    def system_one(
        self,
        state: Any,
        questions: Mapping[str, Any],
        media: Sequence[Any] | None = None,
        *,
        score_readout: str | None = None,
        set_size: int | None = None,
    ) -> dict[str, Any]:
        kwargs = _call_kwargs(score_readout, set_size)
        with self._infer_lock:
            result = self._model.system_one(
                state,
                dict(questions),
                media=list(media) if media else None,
                **kwargs,
            )
        if not isinstance(result, dict):
            raise RuntimeError("model.system_one did not return an object")
        return result

    def system_one_batch(
        self,
        requests: Sequence[tuple[Any, Mapping[str, Any], Sequence[Any] | None]],
        *,
        score_readout: str | None = None,
        set_size: int | None = None,
    ) -> list[dict[str, Any]]:
        tuples: list[tuple[Any, ...]] = []
        for state, questions, media in requests:
            copied = dict(questions)
            if media:
                tuples.append((state, copied, list(media)))
            else:
                tuples.append((state, copied))
        kwargs = _call_kwargs(score_readout, set_size)
        with self._infer_lock:
            results = self._model.system_one_batch(tuples, **kwargs)
        if not isinstance(results, list):
            raise RuntimeError("model.system_one_batch did not return a list")
        return results

    def close(self) -> None:
        self._model = None
        self._processor = None


def _call_kwargs(score_readout: str | None, set_size: int | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if score_readout is not None:
        kwargs["score_readout"] = score_readout
    if set_size is not None:
        kwargs["set_size"] = set_size
    return kwargs
