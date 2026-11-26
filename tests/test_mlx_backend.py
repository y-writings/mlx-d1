import os
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from mlx_d1.backends.mlx import MlxVlmBackend
from mlx_d1.errors import ModelNotAvailable


class _Missing:
    pass


MISSING = _Missing()


class _Model:
    def __init__(self):
        self.one_calls = []
        self.batch_calls = []
        self._in_flight = 0
        self.max_in_flight = 0
        self._lock = threading.Lock()

    def system_one(self, state, questions, media=None, score_readout=MISSING, set_size=MISSING):
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            time.sleep(0.05)
            self.one_calls.append(
                {
                    "state": state,
                    "questions": questions,
                    "media": media,
                    "score_readout": score_readout,
                    "set_size": set_size,
                }
            )
            return {"answers": {}, "usage": {"input_tokens": 1, "output_tokens": 0}}
        finally:
            with self._lock:
                self._in_flight -= 1

    def system_one_batch(self, requests, score_readout=MISSING, set_size=MISSING):
        self.batch_calls.append(
            {"requests": requests, "score_readout": score_readout, "set_size": set_size}
        )
        return [{"answers": {}, "usage": {"input_tokens": 1, "output_tokens": 0}} for _ in requests]


@pytest.fixture
def loaded_model(monkeypatch):
    model = _Model()
    module = types.ModuleType("mlx_vlm")
    module.load = lambda model_id: (model, {"processor": model_id})
    monkeypatch.setitem(sys.modules, "mlx_vlm", module)
    return model


def _checkpoint(path: Path) -> Path:
    path.mkdir()
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "weights.safetensors").write_bytes(b"weights")
    return path


def test_mlx_backend_loads_only_a_local_directory(loaded_model, tmp_path: Path):
    folder = _checkpoint(tmp_path / "d1")
    seen: dict[str, str | None] = {}
    original = sys.modules["mlx_vlm"].load

    def load(model_id: str):
        import httpx
        import huggingface_hub.constants as constants
        from huggingface_hub.errors import OfflineModeIsEnabled
        from huggingface_hub.utils._http import hf_request_event_hook

        seen["path"] = model_id
        seen["offline"] = os.environ.get("HF_HUB_OFFLINE")
        seen["flag"] = str(constants.HF_HUB_OFFLINE)
        request = httpx.Request("GET", "https://huggingface.co/LiquidAI/d1-2B-MLX-8bit")
        try:
            hf_request_event_hook(request)
        except OfflineModeIsEnabled:
            seen["blocked"] = "yes"
        else:
            seen["blocked"] = "no"
        return original(model_id)

    sys.modules["mlx_vlm"].load = load
    backend = MlxVlmBackend("LiquidAI/d1-2B-MLX-8bit", weights_path=folder)
    assert seen["path"] == str(folder.resolve())
    assert seen["offline"] == "1"
    assert seen["flag"] == "True"
    assert seen["blocked"] == "yes"
    assert os.environ.get("HF_HUB_OFFLINE") != "1"
    assert backend.model_id == "LiquidAI/d1-2B-MLX-8bit"
    with pytest.raises(ModelNotAvailable):
        MlxVlmBackend("LiquidAI/d1-2B-MLX-8bit", weights_path=tmp_path / "missing")
    assert loaded_model.one_calls == []


def test_mlx_backend_calls_system_one_and_batches_media(loaded_model, tmp_path: Path):
    backend = MlxVlmBackend("LiquidAI/d1-2B-MLX-8bit", weights_path=_checkpoint(tmp_path / "d1"))
    image = object()
    backend.system_one(
        "ticket", {"refund": {"type": "noul"}}, None, score_readout="mean", set_size=4
    )
    backend.system_one(None, {"cats": {"type": "number"}}, [image])
    backend.system_one_batch(
        [
            ("left", {"q": {"type": "noul"}}, None),
            ("right", {"q": {"type": "noul"}}, [image]),
        ],
        score_readout="argmax",
        set_size=2,
    )
    assert loaded_model.one_calls[0]["media"] is None
    assert loaded_model.one_calls[0]["score_readout"] == "mean"
    assert loaded_model.one_calls[0]["set_size"] == 4
    assert loaded_model.one_calls[1]["media"] == [image]
    assert loaded_model.one_calls[1]["score_readout"] is MISSING
    batch = loaded_model.batch_calls[0]
    assert batch["requests"][0] == ("left", {"q": {"type": "noul"}})
    assert batch["requests"][1] == ("right", {"q": {"type": "noul"}}, [image])
    assert batch["score_readout"] == "argmax"
    backend.close()


def test_load_backend_uses_a_prepared_directory_without_prompt(
    loaded_model, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from mlx_d1.config import Settings
    from mlx_d1.factory import load_backend

    folder = _checkpoint(tmp_path / "ready")

    def refuse(*_args, **_kwargs):
        raise AssertionError("prepare_weights should not run when a directory is already chosen")

    monkeypatch.setattr("mlx_d1.weights.prepare_weights", refuse)
    backend = load_backend(
        Settings(backend="mlx", model="LiquidAI/d1-2B-MLX-8bit"),
        weights_path=folder,
    )
    assert backend.weights_path == folder.resolve()


def test_mlx_backend_lock_is_serial(loaded_model, tmp_path: Path):
    backend = MlxVlmBackend("LiquidAI/d1-2B-MLX-8bit", weights_path=_checkpoint(tmp_path / "d1"))
    threads = [
        threading.Thread(target=lambda: backend.system_one("s", {"q": {}}, None)) for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert loaded_model.max_in_flight == 1
