import pytest

from mlx_d1.errors import QuestionError
from mlx_d1.questions import check_request, validate_questions

NOUL = {"type": "noul", "instructions": "Is this a refund request?"}


def test_passthrough_keeps_unknown_fields_and_choice_shape():
    questions = validate_questions(
        {
            "team": {
                "type": "choice",
                "instructions": "  Which team?  ",
                "criteria": {"billing": "Charges", "shipping": None},
                "multiple": False,
                "note": "kept",
            }
        }
    )
    assert questions["team"]["instructions"] == "Which team?"
    assert questions["team"]["criteria"] == {"billing": "Charges", "shipping": ""}
    assert questions["team"]["multiple"] is False
    assert questions["team"]["note"] == "kept"


def test_bool_is_noul_alias():
    questions = validate_questions({"flag": {"type": "bool", "instructions": "Yes?"}})
    assert questions["flag"]["type"] == "noul"


@pytest.mark.parametrize(
    "questions",
    [
        {},
        {"q": {"type": "spans", "instructions": "Find spans"}},
        {"q": {"type": "noul"}},
        {"q": {"type": "choice", "instructions": "Pick", "criteria": {"only": "one"}}},
        {"q": {"type": "choice", "instructions": "Pick", "criteria": ["a", "b"]}},
        {"q": {"type": "score", "instructions": "Rate", "criteria": ["only"]}},
        {"q": {"type": "score", "instructions": "Rate", "criteria": [str(i) for i in range(11)]}},
        {"q": {"type": "number", "instructions": "How many?", "range": [10, 0], "step": 1}},
        {"q": {"type": "number", "instructions": "How many?", "range": [0, 10], "step": 3}},
        {"q": {"type": "number", "instructions": "How many?", "step": 1}},
        {"q": {"type": "boxes", "instructions": "Every cat", "max": 17}},
        {"q": {"type": "boxes", "instructions": "Every cat", "max": 0}},
    ],
)
def test_rejects_bad_questions(questions):
    with pytest.raises(QuestionError):
        validate_questions(questions)


def test_number_range_and_boxes_max_pass():
    questions = validate_questions(
        {
            "count": {"type": "number", "instructions": "How many?", "range": [0, 10], "step": 2},
            "cats": {"type": "boxes", "instructions": "Every cat", "max": 16},
        }
    )
    assert questions["count"]["range"] == [0, 10]
    assert questions["count"]["step"] == 2
    assert questions["cats"]["max"] == 16


def test_box_requires_exactly_one_image():
    questions = validate_questions({"where": {"type": "box", "instructions": "The remote"}})
    with pytest.raises(QuestionError, match="exactly one image"):
        check_request(questions, state=None, image_count=2)
    check_request(questions, state=None, image_count=1)


def test_text_only_state_is_enough():
    questions = validate_questions({"q": NOUL})
    check_request(questions, state="hello", image_count=0)
    with pytest.raises(QuestionError):
        check_request(questions, state=None, image_count=0)
