"""Validate d1 questions without adding use-case content.

Unknown question fields are kept and forwarded. The server checks the fields
the model card defines and does not invent criteria, options, or cutoffs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from mlx_d1.errors import QuestionError

QUESTION_TYPES = ("noul", "choice", "score", "text", "number", "date", "box", "boxes")
_BOX_TYPES = {"box", "boxes"}


def validate_questions(questions: Any, *, max_questions: int = 128) -> dict[str, dict[str, Any]]:
    """Return a JSON-ready copy of ``questions`` or raise ``QuestionError``."""
    if not isinstance(questions, dict) or not questions:
        raise QuestionError("questions must be a non-empty object")
    if len(questions) > max_questions:
        raise QuestionError(f"at most {max_questions} questions are allowed")
    normalized: dict[str, dict[str, Any]] = {}
    for name, raw in questions.items():
        if not isinstance(name, str) or not name.strip():
            raise QuestionError("question names must be non-empty strings")
        if name != name.strip() or "\n" in name or "\r" in name:
            raise QuestionError(
                f"{name!r}: question names must be a single line without surrounding space"
            )
        if len(name) > 128:
            raise QuestionError(f"{name}: question name is longer than 128 characters")
        normalized[name] = _validate_one(name, raw)
    return normalized


def check_request(questions: Mapping[str, Any], *, state: Any, image_count: int) -> None:
    """Check constraints that depend on the state and the images together."""
    if state is None and image_count == 0:
        raise QuestionError("provide a state, at least one image, or both")
    box_names = [name for name, question in questions.items() if question.get("type") in _BOX_TYPES]
    if box_names and image_count != 1:
        joined = ", ".join(box_names)
        raise QuestionError(f"{joined}: box and boxes questions need exactly one image")


def ensure_state_size(state: Any, limit: int) -> None:
    """Reject a state that is not JSON or that exceeds ``limit`` bytes."""
    if state is None:
        return
    try:
        if isinstance(state, str):
            size = len(state.encode("utf-8"))
        else:
            size = len(json.dumps(state, ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise QuestionError("state must be a JSON value") from exc
    if size > limit:
        raise QuestionError(f"state is larger than max_state_bytes ({limit})")


def _validate_one(name: str, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise QuestionError(f"{name}: question must be an object")
    question = dict(raw)
    qtype = question.get("type")
    if qtype == "bool":
        # mlx-vlm's HTTP server accepts bool as another spelling of noul.
        qtype = "noul"
        question["type"] = "noul"
    if qtype not in QUESTION_TYPES:
        supported = ", ".join(QUESTION_TYPES)
        raise QuestionError(
            f"{name}: unsupported type {qtype!r}. Supported types: {supported}. "
            "bool is accepted as an alias of noul"
        )
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise QuestionError(f"{name}: instructions must be a non-empty string")
    question["instructions"] = instructions.strip()
    if qtype == "choice":
        question["criteria"] = _choice_criteria(name, question.get("criteria"))
        if "multiple" in question and not isinstance(question["multiple"], bool):
            raise QuestionError(f"{name}: multiple must be a boolean")
    elif qtype == "score":
        question["criteria"] = _score_criteria(name, question.get("criteria"))
    elif qtype == "number":
        _check_number(name, question)
    elif qtype == "boxes" and "max" in question and question["max"] is not None:
        maximum = question["max"]
        if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 16:
            raise QuestionError(f"{name}: max must be an integer from 1 to 16")
    return question


def _choice_criteria(name: str, criteria: Any) -> dict[str, str]:
    if not isinstance(criteria, dict) or len(criteria) < 2:
        raise QuestionError(f"{name}: choice criteria must be an object of at least two options")
    normalized: dict[str, str] = {}
    for key, description in criteria.items():
        if not isinstance(key, str) or not key.strip() or key != key.strip():
            raise QuestionError(f"{name}: option names must be non-empty strings")
        if description is None:
            description = ""
        if not isinstance(description, str):
            raise QuestionError(f"{name}: option {key!r} description must be a string or null")
        normalized[key] = description
    return normalized


def _score_criteria(name: str, criteria: Any) -> list[str]:
    if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
        raise QuestionError(
            f"{name}: score criteria must be a list of 2 to 10 levels, lowest first"
        )
    if not all(isinstance(level, str) and level.strip() for level in criteria):
        raise QuestionError(f"{name}: score levels must be non-empty strings")
    return [level.strip() for level in criteria]


def _check_number(name: str, question: dict[str, Any]) -> None:
    if "unit" in question and question["unit"] is not None:
        unit = question["unit"]
        if not isinstance(unit, str) or not unit.strip():
            raise QuestionError(f"{name}: unit must be a non-empty string or null")
        question["unit"] = unit.strip()
    has_range = "range" in question and question["range"] is not None
    has_step = "step" in question and question["step"] is not None
    if has_step and not has_range:
        raise QuestionError(f"{name}: step requires range")
    if not has_range:
        return
    bounds = question["range"]
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or not all(_is_number(value) for value in bounds)
    ):
        raise QuestionError(f"{name}: range must be [low, high]")
    low, high = bounds
    if not low < high:
        raise QuestionError(f"{name}: range low must be less than high")
    question["range"] = [low, high]
    if not has_step:
        return
    step = question["step"]
    if not _is_number(step) or step <= 0:
        raise QuestionError(f"{name}: step must be a positive number")
    quot = (high - low) / step
    if abs(quot - round(quot)) > 1e-6 * max(1.0, abs(quot)):
        raise QuestionError(f"{name}: step must divide the range")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
