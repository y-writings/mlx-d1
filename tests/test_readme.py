import json
import re

from fastapi.testclient import TestClient

from mlx_d1.backends.fake import FakeBackend
from mlx_d1.config import load_settings
from mlx_d1.server import create_app
from tests.conftest import ROOT

QUESTION_TYPES = {"noul", "choice", "score", "text", "number", "date", "box", "boxes"}


def _json_blocks() -> list[object]:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    return [
        json.loads(match.group(1)) for match in re.finditer(r"```json\n(.*?)```", text, re.DOTALL)
    ]


def test_readme_examples_cover_every_question_type_and_are_accepted():
    settings = load_settings(config_path=ROOT / "examples" / "config.toml", environ={})
    app = create_app(settings, FakeBackend())
    blocks = _json_blocks()
    assert blocks
    seen_types: set[str] = set()
    saw_multiple = False
    saw_range = False
    with TestClient(app) as client:
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if "questions" in block and "answers" not in block:
                response = client.post("/v1/systemone", json=block)
                assert response.status_code == 200, response.text
                for question in block["questions"].values():
                    seen_types.add(question["type"])
                    saw_multiple = saw_multiple or question.get("multiple") is True
                    saw_range = saw_range or "range" in question
            elif {"a", "criteria"} <= block.keys() and "winner" not in block:
                response = client.post("/v1/compare", json=block)
                assert response.status_code == 200, response.text
            elif "requests" in block and "results" not in block:
                response = client.post("/v1/systemone/batch", json=block)
                assert response.status_code == 200, response.text
            elif (
                "images" in block
                and "mcpServers" not in block
                and "questions" not in block
                and "answers" not in block
            ):
                response = client.post("/v1/presets/image_artifacts", json=block)
                assert response.status_code == 200, response.text
    assert seen_types == QUESTION_TYPES
    assert saw_multiple
    assert saw_range
