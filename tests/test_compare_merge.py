from mlx_d1.backends.fake import choice_answer
from mlx_d1.compare import decide_winner, merge_criteria, normalize_criteria

FORWARD = ("a", "b")
REVERSE = ("b", "a")


def _answers(choice: str) -> dict:
    return {"preference": choice_answer(choice)}


def test_side_that_wins_both_orders_wins():
    merged = merge_criteria(["preference"], _answers("FIRST"), _answers("SECOND"), FORWARD, REVERSE)
    assert merged["winner"] == "a"
    assert merged["criteria"]["preference"]["forward"]["mapped"] == "a"
    assert merged["criteria"]["preference"]["reversed"]["mapped"] == "a"


def test_split_is_a_tie_and_probabilities_are_not_averaged():
    merged = merge_criteria(["preference"], _answers("FIRST"), _answers("FIRST"), FORWARD, REVERSE)
    assert merged["winner"] == "tie"
    assert merged["criteria"]["preference"]["forward"]["probabilities"] == {
        "FIRST": 0.8,
        "SECOND": 0.2,
    }
    assert merged["criteria"]["preference"]["reversed"]["mapped"] == "b"


def test_majority_of_criteria():
    forward = {
        "one": choice_answer("FIRST"),
        "two": choice_answer("SECOND"),
        "three": choice_answer("FIRST"),
    }
    reverse = {
        "one": choice_answer("SECOND"),
        "two": choice_answer("FIRST"),
        "three": choice_answer("SECOND"),
    }
    merged = merge_criteria(["one", "two", "three"], forward, reverse, FORWARD, REVERSE)
    assert merged["criteria"]["one"]["winner"] == "a"
    assert merged["criteria"]["two"]["winner"] == "b"
    assert merged["winner"] == "a"
    assert merged["counts"] == {"a": 2, "b": 1, "tie": 0}


def test_unexpected_choice_is_a_tie():
    assert decide_winner("a", None) == "tie"
    merged = merge_criteria(
        ["preference"],
        {"preference": {"type": "choice", "choice": "maybe"}},
        _answers("SECOND"),
        FORWARD,
        REVERSE,
    )
    assert merged["winner"] == "tie"
    assert "error" in merged["criteria"]["preference"]["forward"]


def test_normalize_criteria_shapes():
    assert normalize_criteria("Which is better?") == [
        {"name": "preference", "instructions": "Which is better?"}
    ]
    assert normalize_criteria(["First angle", "Second angle"])[1]["name"] == "criterion_2"
    named = normalize_criteria([{"name": "sharpness", "instructions": "Which is sharper?"}])
    assert named[0]["name"] == "sharpness"
