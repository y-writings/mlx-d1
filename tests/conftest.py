import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlx_d1.backends.fake import FakeBackend
from mlx_d1.config import Settings
from mlx_d1.server import create_app

ROOT = Path(__file__).resolve().parents[1]
PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)
PNG_BYTES = base64.b64decode(PNG_BASE64)
DATA_URL = f"data:image/png;base64,{PNG_BASE64}"


@pytest.fixture
def backend():
    return FakeBackend(model_id="fake-d1")


@pytest.fixture
def app(backend):
    return create_app(Settings(backend="fake", model="fake-d1"), backend)


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client
