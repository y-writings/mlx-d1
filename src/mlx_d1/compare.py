"""A/B comparison with a position swap.

The caller supplies both candidates and every criterion. This module only
labels the two positions FIRST and SECOND, asks the same questions in both
orders, and merges the mapped winners. A side must win both orders. A split,
or an answer that is not FIRST or SECOND, is a tie. Probabilities are reported
for each order and are not averaged, and no cutoff is applied.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mlx_d1.errors import QuestionError

Order = tuple[str, str]
_FIRST = {"FIRST", "first", "First"}
_SECOND = {"SECOND", "second", "Second"}


@dataclass(frozen=True)
class PreparedCall:
    order: Order
    state: str
    questions: dict[str, dict[str, Any]]
    media: list[Any] | None


def normalize_criteria(criteria: Any) -> list[dict[str, str]]:
    """Accept one instruction, a list of instructions, or ``{name, instructions}`` objects."""
    if isinstance(criteria, str):
        items: list[Any] = [criteria]
    elif isinstance(criteria, list):
        items = criteria
    else:
        raise QuestionError("criteria must be a string or a list")
    if not items:
        raise QuestionError("criteria must not be empty")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(items, start=1):
        if isinstance(item, str):
            name = "preference" if len(items) == 1 else f"criterion_{index}"
            instructions = item
        elif isinstance(item, dict):
            extra = set(item) - {"name", "instructions"}
            if extra:
                joined = ", ".join(sorted(str(key) for key in extra))
                raise QuestionError(f"criterion has unsupported fields: {joined}")
            name = item.get("name")
            instructions = item.get("instructions")
            if not isinstance(name, str):
                raise QuestionError("criterion name must be a non-empty string")
        else:
            raise QuestionError(
                "each criterion must be a string or an object with name and instructions"
            )
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise QuestionError("criterion name must be a non-empty string")
        if "\n" in name or "\r" in name or len(name) > 128:
            raise QuestionError(
                f"{name!r}: criterion name must be a single line of at most 128 characters"
            )
        if not isinstance(instructions, str) or not instructions.strip():
            raise QuestionError(f"{name}: instructions must be a non-empty string")
        if name in seen:
            raise QuestionError(f"duplicate criterion name: {name}")
        seen.add(name)
        normalized.append({"name": name, "instructions": instructions.strip()})
    return normalized


def build_calls(
    *,
    a_text: str | None,
    b_text: str | None,
    a_images: list[Any],
    b_images: list[Any],
    shared: Any,
    criteria: list[dict[str, str]],
) -> list[PreparedCall]:
    """Build the ``[A, B]`` call and then the ``[B, A]`` call."""
    questions = comparison_questions(criteria)
    calls: list[PreparedCall] = []
    sides = {
        "a": (a_text, a_images),
        "b": (b_text, b_images),
    }
    for order in (("a", "b"), ("b", "a")):
        first_text, first_images = sides[order[0]]
        second_text, second_images = sides[order[1]]
        media = [*first_images, *second_images]
        calls.append(
            PreparedCall(
                order=order,
                state=render_comparison_state(
                    shared,
                    first_text,
                    second_text,
                    len(first_images),
                    len(second_images),
                ),
                questions={key: dict(value) for key, value in questions.items()},
                media=media or None,
            )
        )
    return calls


def comparison_questions(criteria: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """Turn caller criteria into FIRST/SECOND choice questions."""
    questions: dict[str, dict[str, Any]] = {}
    for item in criteria:
        questions[item["name"]] = {
            "type": "choice",
            "instructions": (
                f"{item['instructions'].rstrip()} Answer FIRST if FIRST is the one to choose, "
                "or SECOND if SECOND is the one to choose."
            ),
            "criteria": {
                "FIRST": "The candidate labeled FIRST.",
                "SECOND": "The candidate labeled SECOND.",
            },
        }
    return questions


def render_comparison_state(
    shared: Any,
    first_text: str | None,
    second_text: str | None,
    first_images: int,
    second_images: int,
) -> str:
    """Describe the two positions. The model never sees the names ``a`` and ``b``."""
    blocks: list[str] = []
    shared_text = render_shared(shared)
    if shared_text:
        blocks.append("Context:\n" + shared_text)
    blocks.append(
        "\n".join(
            [
                "Two candidates are labeled FIRST and SECOND.",
                "Any images are attached in order: every FIRST image, then every SECOND image.",
                "Choose the candidate the question asks for.",
                "Do not prefer a candidate only because of its position.",
            ]
        )
    )
    blocks.append(f"FIRST ({_image_phrase(first_images)}):\n{first_text or '(no text)'}")
    blocks.append(f"SECOND ({_image_phrase(second_images)}):\n{second_text or '(no text)'}")
    return "\n\n".join(blocks)


def render_shared(shared: Any) -> str | None:
    if shared is None:
        return None
    if isinstance(shared, str):
        text = shared.strip()
        return text or None
    try:
        return json.dumps(shared, ensure_ascii=False, indent=2)
    except (TypeError, ValueError) as exc:
        raise QuestionError("state must be a JSON value") from exc


def merge_criteria(
    names: list[str],
    forward_answers: dict[str, Any],
    reverse_answers: dict[str, Any],
    forward_order: Order,
    reverse_order: Order,
) -> dict[str, Any]:
    """Merge the two orders. The overall winner is a strict majority of criteria."""
    per_criterion: dict[str, Any] = {}
    winners: list[str] = []
    for name in names:
        forward = interpret_pass(forward_answers.get(name), forward_order)
        reverse = interpret_pass(reverse_answers.get(name), reverse_order)
        winner = decide_winner(forward.get("mapped"), reverse.get("mapped"))
        winners.append(winner)
        per_criterion[name] = {"winner": winner, "forward": forward, "reversed": reverse}
    counts = {"a": winners.count("a"), "b": winners.count("b"), "tie": winners.count("tie")}
    if counts["a"] > counts["b"]:
        overall = "a"
    elif counts["b"] > counts["a"]:
        overall = "b"
    else:
        overall = "tie"
    return {"winner": overall, "counts": counts, "criteria": per_criterion}


def interpret_pass(answer: Any, order: Order) -> dict[str, Any]:
    """Map one order's answer back onto candidate ``a`` or ``b``."""
    base = {
        "order": list(order),
        "choice": None,
        "mapped": None,
        "probabilities": None,
        "confidence": None,
        "answer": answer,
    }
    if not isinstance(answer, dict):
        base["error"] = "missing answer"
        return base
    choice = answer.get("choice")
    if not isinstance(choice, str):
        probabilities = answer.get("probabilities")
        if isinstance(probabilities, dict) and probabilities:
            best = max(probabilities, key=lambda key: probabilities[key])
            if isinstance(best, str):
                choice = best
    mapped = map_label(choice if isinstance(choice, str) else None, order)
    probabilities = answer.get("probabilities")
    confidence = answer.get("confidence")
    base.update(
        {
            "choice": choice if isinstance(choice, str) else None,
            "mapped": mapped,
            "probabilities": probabilities if isinstance(probabilities, dict) else None,
            "confidence": confidence if _is_number(confidence) else None,
        }
    )
    if mapped is None:
        base["error"] = "expected a FIRST or SECOND choice"
    return base


def map_label(choice: str | None, order: Order) -> str | None:
    if choice in _FIRST:
        return order[0]
    if choice in _SECOND:
        return order[1]
    return None


def decide_winner(mapped_forward: str | None, mapped_reverse: str | None) -> str:
    if mapped_forward is None or mapped_reverse is None:
        return "tie"
    if mapped_forward == mapped_reverse:
        return mapped_forward
    return "tie"


def _image_phrase(count: int) -> str:
    if count == 0:
        return "no images"
    if count == 1:
        return "1 image"
    return f"{count} images"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
