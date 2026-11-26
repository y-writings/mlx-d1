import sys
import threading
import time
import types

import pytest

from mlx_d1.backends.mlx import MlxVlmBackend


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


def test_mlx_backend_calls_system_one_and_batches_media(loaded_model):
    backend = MlxVlmBackend("LiquidAI/d1-2B-MLX-8bit")
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


def test_mlx_backend_lock_is_serial(loaded_model):
    backend = MlxVlmBackend("LiquidAI/d1-2B-MLX-8bit")
    threads = [
        threading.Thread(target=lambda: backend.system_one("s", {"q": {}}, None)) for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert loaded_model.max_in_flight == 1
