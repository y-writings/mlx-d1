from mlx_d1.backends.fake import FakeBackend, choice_answer
from mlx_d1.config import load_settings
from mlx_d1.server import create_app
from tests.conftest import DATA_URL, ROOT

TICKET = "I was charged twice and my parcel never arrived. Please refund one charge."


def test_health_reports_fake_backend_without_weights(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["backend"] == "fake"
    assert body["model_loaded"] is False
    assert body["weights_loaded"] is False
    assert body["model"] == "fake-d1"


def test_text_only_request_reaches_system_one_unchanged(client, backend):
    questions = {
        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "Charges", "shipping": "Delivery"},
        },
    }
    response = client.post("/v1/systemone", json={"state": TICKET, "questions": questions})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"model", "answers", "usage", "timing"}
    assert body["usage"]["output_tokens"] == 0
    assert body["answers"]["refund"]["type"] == "noul"
    assert body["timing"]["inference_ms"] >= 0
    call = backend.calls[0]
    assert call["kind"] == "one"
    assert call["state"] == TICKET
    assert call["media"] is None
    assert call["questions"] == questions
    assert call["score_readout"] == "median"


def test_request_can_override_readout_and_set_size(client, backend):
    response = client.post(
        "/v1/systemone",
        json={
            "state": TICKET,
            "questions": {"refund": {"type": "noul", "instructions": "Refund?"}},
            "score_readout": "mean",
            "set_size": 4,
        },
    )
    assert response.status_code == 200
    assert backend.calls[0]["score_readout"] == "mean"
    assert backend.calls[0]["set_size"] == 4


def test_each_question_type_is_accepted(client):
    response = client.post(
        "/v1/systemone",
        json={
            "state": TICKET,
            "images": [DATA_URL],
            "questions": {
                "refund": {"type": "noul", "instructions": "Is this a refund request?"},
                "team": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"billing": "Charges", "shipping": "Delivery"},
                },
                "topics": {
                    "type": "choice",
                    "multiple": True,
                    "instructions": "Which topics?",
                    "criteria": {"billing": "Charges", "delivery": "Parcels"},
                },
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent?",
                    "criteria": ["low", "medium", "high"],
                },
                "payee": {"type": "text", "instructions": "Who is named?"},
                "total": {"type": "number", "instructions": "What amount is named?"},
                "due": {"type": "date", "instructions": "Which date is named?"},
                "count": {
                    "type": "number",
                    "instructions": "How many items are visible?",
                    "range": [0, 10],
                    "step": 1,
                },
                "remote": {"type": "box", "instructions": "Where is the remote?"},
                "cats": {"type": "boxes", "instructions": "Every cat.", "max": 4},
            },
        },
    )
    assert response.status_code == 200
    answers = response.json()["answers"]
    assert answers["topics"]["type"] == "choice"
    assert "choices" in answers["topics"]
    assert answers["count"]["interval"] == [0, 0]
    assert answers["remote"]["type"] == "box"
    assert answers["cats"]["type"] == "boxes"


def test_invalid_question_does_not_call_the_model(client, backend):
    response = client.post(
        "/v1/systemone",
        json={
            "state": TICKET,
            "questions": {"q": {"type": "score", "instructions": "Rate", "criteria": ["only"]}},
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "validation_error"
    assert backend.calls == []


def test_path_outside_allowlist_is_rejected(client, backend, tmp_path):
    image = tmp_path / "pixel.png"
    image.write_bytes(b"not-read")
    response = client.post(
        "/v1/systemone",
        json={
            "state": "photo",
            "images": [str(image)],
            "questions": {"q": {"type": "noul", "instructions": "Is there a photo?"}},
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "image_error"
    assert backend.calls == []


def test_box_without_an_image_is_rejected(client, backend):
    response = client.post(
        "/v1/systemone",
        json={
            "state": "no photo",
            "questions": {"where": {"type": "box", "instructions": "Where?"}},
        },
    )
    assert response.status_code == 400
    assert backend.calls == []


def test_batch_is_one_model_call(client, backend):
    response = client.post(
        "/v1/systemone/batch",
        json={
            "requests": [
                {"state": "one", "questions": {"q": {"type": "noul", "instructions": "Yes?"}}},
                {"state": "two", "questions": {"q": {"type": "noul", "instructions": "Yes?"}}},
            ],
            "set_size": 3,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 2
    assert backend.calls[0]["kind"] == "batch"
    assert len(backend.calls[0]["requests"]) == 2
    assert backend.calls[0]["set_size"] == 3
    assert body["usage"]["input_tokens"] == 2


def test_compare_cancels_a_position_bias(client, backend):
    response = client.post(
        "/v1/compare",
        json={
            "a": {"text": "alpha"},
            "b": {"text": "beta"},
            "criteria": [{"name": "preference", "instructions": "Which is better?"}],
            "state": "Shared brief",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["winner"] == "tie"
    assert body["criteria"]["preference"]["forward"]["mapped"] == "a"
    assert body["criteria"]["preference"]["reversed"]["mapped"] == "b"
    call = backend.calls[0]
    assert call["kind"] == "batch"
    forward, reverse = call["requests"]
    assert forward["state"].index("alpha") < forward["state"].index("beta")
    assert reverse["state"].index("beta") < reverse["state"].index("alpha")
    assert "Which is better?" in forward["questions"]["preference"]["instructions"]
    assert forward["questions"]["preference"]["criteria"] == {
        "FIRST": "The candidate labeled FIRST.",
        "SECOND": "The candidate labeled SECOND.",
    }
    assert forward["media"] is None


def test_compare_keeps_the_side_that_wins_both_orders():
    def responder(state, questions, media):
        del media
        text = state if isinstance(state, str) else ""
        first = text[text.rfind("\nFIRST (") : text.rfind("\nSECOND (")]
        second = text[text.rfind("\nSECOND (") :]
        if "WINNER" in first and "WINNER" not in second:
            choice = "FIRST"
        elif "WINNER" in second and "WINNER" not in first:
            choice = "SECOND"
        else:
            choice = "FIRST"
        return {
            "answers": {name: choice_answer(choice) for name in questions},
            "usage": {"input_tokens": 5, "output_tokens": 0},
        }

    backend = FakeBackend(responder=responder)
    app = create_app(load_settings(environ={}), backend)
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.post(
            "/v1/compare",
            json={
                "a": {"text": "alpha WINNER", "images": [DATA_URL]},
                "b": {"text": "beta", "images": [DATA_URL]},
                "criteria": ["Which should we keep?"],
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["winner"] == "a"
    assert body["usage"]["input_tokens"] == 10
    forward, reverse = backend.calls[0]["requests"]
    assert forward["media"][0] is reverse["media"][1]
    assert forward["media"][1] is reverse["media"][0]


def test_repeated_requests_keep_returning_json(client):
    payload = {"state": "again", "questions": {"q": {"type": "noul", "instructions": "Still up?"}}}
    first = client.post("/v1/systemone", json=payload)
    second = client.post("/v1/systemone", json=payload)
    assert first.status_code == second.status_code == 200
    assert first.json()["answers"]["q"]["type"] == "noul"
    assert second.json()["answers"]["q"]["type"] == "noul"


def test_preset_round_trip(tmp_path):
    settings = load_settings(config_path=ROOT / "examples" / "config.toml", environ={})
    backend = FakeBackend()
    app = create_app(settings, backend)
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        listed = client.get("/v1/presets").json()
        assert listed["presets"][0]["name"] == "image_artifacts"
        missing = client.post("/v1/presets/missing", json={})
        assert missing.status_code == 404
        response = client.post(
            "/v1/presets/image_artifacts", json={"state": None, "images": [DATA_URL]}
        )
    assert response.status_code == 200
    answers = response.json()["answers"]
    assert set(answers) == {"extra_limbs", "warped_text", "warped_geometry", "severity"}
    sent = backend.calls[0]["questions"]
    assert "exposure" in sent["severity"]["instructions"]
    assert backend.calls[0]["media"] is not None
