"""Prepare requests and run them through one serial inference gate."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mlx_d1.backends.base import DecisionBackend
from mlx_d1.compare import build_calls, merge_criteria, normalize_criteria
from mlx_d1.config import Settings
from mlx_d1.errors import ImageError, QuestionError, RequestError
from mlx_d1.images import load_image_inputs
from mlx_d1.questions import check_request, ensure_state_size, validate_questions

log = logging.getLogger("mlx_d1")


class InferenceGate:
    """Run model calls one at a time.

    The lock is held across the thread that executes the call, so two requests
    cannot enter MLX together. Image decoding happens before the lock.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def run(self, func: Callable[[], Any]) -> Any:
        async with self._lock:
            return await asyncio.to_thread(func)


@dataclass(frozen=True)
class Prepared:
    state: Any
    questions: dict[str, dict[str, Any]]
    media: list[Any] | None


class JudgeService:
    def __init__(self, settings: Settings, backend: DecisionBackend, gate: InferenceGate) -> None:
        self.settings = settings
        self.backend = backend
        self.gate = gate

    async def system_one(
        self,
        *,
        state: Any,
        images: list[str] | None,
        questions: Any,
        score_readout: str | None,
        set_size: int | None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        prepared = await self._prepare(state, images, questions)
        result, inference_ms = await self._infer_one(
            prepared, score_readout=score_readout, set_size=set_size
        )
        answers, usage = split_result(result)
        latency_ms = _elapsed_ms(started)
        log.info(
            "systemone questions=%d images=%d latency_ms=%s inference_ms=%s",
            len(answers),
            0 if prepared.media is None else len(prepared.media),
            latency_ms,
            inference_ms,
        )
        return _envelope(self.backend.model_id, answers, usage, latency_ms, inference_ms)

    async def system_one_batch(
        self,
        items: list[dict[str, Any]],
        *,
        score_readout: str | None,
        set_size: int | None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if not items:
            raise RequestError(400, "validation_error", "requests must not be empty")
        if len(items) > self.settings.max_batch:
            raise RequestError(
                400,
                "validation_error",
                f"at most {self.settings.max_batch} requests are allowed in one batch",
            )
        prepared: list[Prepared] = []
        for index, item in enumerate(items, start=1):
            try:
                prepared.append(
                    await self._prepare(
                        item.get("state"), item.get("images"), item.get("questions")
                    )
                )
            except RequestError as exc:
                raise RequestError(exc.status, exc.code, f"request {index}: {exc}") from exc
        results, inference_ms = await self._infer_batch(
            prepared, score_readout=score_readout, set_size=set_size
        )
        bodies = []
        total_in = 0
        total_out = 0
        for result in results:
            answers, usage = split_result(result)
            usage_body = usage_dict(usage)
            total_in += usage_body["input_tokens"]
            total_out += usage_body["output_tokens"]
            bodies.append({"answers": answers, "usage": usage_body})
        latency_ms = _elapsed_ms(started)
        log.info(
            "batch size=%d latency_ms=%s inference_ms=%s", len(items), latency_ms, inference_ms
        )
        return {
            "model": self.backend.model_id,
            "results": bodies,
            "usage": {"input_tokens": total_in, "output_tokens": total_out},
            "timing": {"latency_ms": latency_ms, "inference_ms": inference_ms},
        }

    async def compare(
        self,
        *,
        a: dict[str, Any],
        b: dict[str, Any],
        criteria: Any,
        state: Any,
        score_readout: str | None,
        set_size: int | None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            normalized = normalize_criteria(criteria)
            if len(normalized) > self.settings.max_questions:
                raise QuestionError(f"at most {self.settings.max_questions} criteria are allowed")
            a_text = _clean_text(a.get("text"), "candidate a")
            b_text = _clean_text(b.get("text"), "candidate b")
            a_specs = list(a.get("images") or [])
            b_specs = list(b.get("images") or [])
            if a_text is None and not a_specs:
                raise QuestionError("candidate a needs text, images, or both")
            if b_text is None and not b_specs:
                raise QuestionError("candidate b needs text, images, or both")
            if len(a_specs) + len(b_specs) > self.settings.max_images:
                raise QuestionError(f"at most {self.settings.max_images} images are allowed")
            ensure_state_size(state, self.settings.max_state_bytes)
            ensure_state_size(a_text, self.settings.max_state_bytes)
            ensure_state_size(b_text, self.settings.max_state_bytes)
        except QuestionError as exc:
            raise RequestError(400, "validation_error", str(exc)) from exc
        try:
            a_images = await self._load(a_specs)
            b_images = await self._load(b_specs)
        except ImageError as exc:
            raise RequestError(400, "image_error", str(exc)) from exc
        calls = build_calls(
            a_text=a_text,
            b_text=b_text,
            a_images=a_images,
            b_images=b_images,
            shared=state,
            criteria=normalized,
        )
        prepared = [Prepared(call.state, call.questions, call.media) for call in calls]
        results, inference_ms = await self._infer_batch(
            prepared, score_readout=score_readout, set_size=set_size
        )
        forward_answers, forward_usage = split_result(results[0])
        reverse_answers, reverse_usage = split_result(results[1])
        merged = merge_criteria(
            [item["name"] for item in normalized],
            forward_answers,
            reverse_answers,
            calls[0].order,
            calls[1].order,
        )
        latency_ms = _elapsed_ms(started)
        log.info(
            "compare criteria=%d winner=%s latency_ms=%s",
            len(normalized),
            merged["winner"],
            latency_ms,
        )
        return {
            "model": self.backend.model_id,
            **merged,
            "usage": add_usage(forward_usage, reverse_usage),
            "timing": {"latency_ms": latency_ms, "inference_ms": inference_ms},
        }

    async def run_preset(
        self,
        name: str,
        *,
        state: Any,
        images: list[str] | None,
        score_readout: str | None,
        set_size: int | None,
    ) -> dict[str, Any]:
        preset = self.settings.presets.get(name)
        if preset is None:
            known = ", ".join(self.settings.presets) or "(none)"
            raise RequestError(
                404, "preset_not_found", f"unknown preset {name!r}. Known presets: {known}"
            )
        return await self.system_one(
            state=state,
            images=images,
            questions=preset.questions,
            score_readout=score_readout,
            set_size=set_size,
        )

    def list_presets(self) -> dict[str, Any]:
        return {
            "presets": [
                {
                    "name": preset.name,
                    "description": preset.description,
                    "questions": preset.questions,
                }
                for preset in self.settings.presets.values()
            ]
        }

    def get_preset(self, name: str) -> dict[str, Any]:
        preset = self.settings.presets.get(name)
        if preset is None:
            known = ", ".join(self.settings.presets) or "(none)"
            raise RequestError(
                404, "preset_not_found", f"unknown preset {name!r}. Known presets: {known}"
            )
        return {
            "name": preset.name,
            "description": preset.description,
            "questions": preset.questions,
        }

    async def _prepare(self, state: Any, images: list[str] | None, questions: Any) -> Prepared:
        try:
            return await asyncio.to_thread(self._prepare_sync, state, images, questions)
        except QuestionError as exc:
            raise RequestError(400, "validation_error", str(exc)) from exc
        except ImageError as exc:
            raise RequestError(400, "image_error", str(exc)) from exc

    def _prepare_sync(self, state: Any, images: list[str] | None, questions: Any) -> Prepared:
        normalized = validate_questions(questions, max_questions=self.settings.max_questions)
        specs = list(images or [])
        if len(specs) > self.settings.max_images:
            raise QuestionError(f"at most {self.settings.max_images} images are allowed")
        ensure_state_size(state, self.settings.max_state_bytes)
        check_request(normalized, state=state, image_count=len(specs))
        media = self._load_sync(specs) if specs else None
        return Prepared(state=state, questions=normalized, media=media)

    async def _load(self, specs: list[str]) -> list[Any]:
        if not specs:
            return []
        return await asyncio.to_thread(self._load_sync, specs)

    def _load_sync(self, specs: list[str]) -> list[Any]:
        return load_image_inputs(
            specs,
            allowed_dirs=self.settings.allowed_image_dirs,
            max_bytes=self.settings.max_image_bytes,
            max_pixels=self.settings.max_image_pixels,
        )

    async def _infer_one(
        self,
        prepared: Prepared,
        *,
        score_readout: str | None,
        set_size: int | None,
    ) -> tuple[Any, float]:
        readout = self._readout(score_readout)
        size = self._set_size(set_size)

        def run() -> tuple[Any, float]:
            started = time.perf_counter()
            result = self.backend.system_one(
                prepared.state,
                prepared.questions,
                prepared.media,
                score_readout=readout,
                set_size=size,
            )
            return result, _elapsed_ms(started)

        return await self._guard(run)

    async def _infer_batch(
        self,
        prepared: list[Prepared],
        *,
        score_readout: str | None,
        set_size: int | None,
    ) -> tuple[list[Any], float]:
        readout = self._readout(score_readout)
        size = self._set_size(set_size)

        def run() -> tuple[list[Any], float]:
            started = time.perf_counter()
            results = self.backend.system_one_batch(
                [(item.state, item.questions, item.media) for item in prepared],
                score_readout=readout,
                set_size=size,
            )
            return results, _elapsed_ms(started)

        results, inference_ms = await self._guard(run)
        if not isinstance(results, list) or len(results) != len(prepared):
            raise RequestError(
                500, "inference_error", "backend returned the wrong number of results"
            )
        return results, inference_ms

    async def _guard(self, func: Callable[[], Any]) -> Any:
        try:
            return await self.gate.run(func)
        except RequestError:
            raise
        except Exception as exc:
            log.exception("inference failed")
            raise RequestError(500, "inference_error", f"{type(exc).__name__}: {exc}") from exc

    def _readout(self, score_readout: str | None) -> str:
        return score_readout or self.settings.score_readout

    def _set_size(self, set_size: int | None) -> int | None:
        return self.settings.set_size if set_size is None else set_size


def split_result(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
        raise RequestError(500, "inference_error", "backend result is missing answers")
    usage = result.get("usage") or {}
    if not isinstance(usage, dict):
        raise RequestError(500, "inference_error", "backend usage is not an object")
    return result["answers"], usage


def usage_dict(usage: dict[str, Any]) -> dict[str, int]:
    return {
        "input_tokens": _token_count(usage.get("input_tokens")),
        "output_tokens": _token_count(usage.get("output_tokens")),
    }


def add_usage(left: dict[str, Any], right: dict[str, Any]) -> dict[str, int]:
    first = usage_dict(left)
    second = usage_dict(right)
    return {
        "input_tokens": first["input_tokens"] + second["input_tokens"],
        "output_tokens": first["output_tokens"] + second["output_tokens"],
    }


def _envelope(
    model_id: str,
    answers: dict[str, Any],
    usage: dict[str, Any],
    latency_ms: float,
    inference_ms: float,
) -> dict[str, Any]:
    return {
        "model": model_id,
        "answers": answers,
        "usage": usage_dict(usage),
        "timing": {"latency_ms": latency_ms, "inference_ms": inference_ms},
    }


def _token_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def _clean_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise QuestionError(f"{label} text must be a string")
    text = value.strip()
    return text or None
