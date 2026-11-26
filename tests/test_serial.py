import asyncio

from mlx_d1.backends.fake import FakeBackend
from mlx_d1.config import Preset, Settings
from mlx_d1.service import InferenceGate, JudgeService

NOUL = {"ready": {"type": "noul", "instructions": "Is it ready?"}}


def test_gate_keeps_model_calls_serial():
    backend = FakeBackend(delay_s=0.05)
    preset = Preset(
        "sample", "example", {"ready": {"type": "noul", "instructions": "Is it ready?"}}
    )
    service = JudgeService(
        Settings(backend="fake", presets={"sample": preset}), backend, InferenceGate()
    )

    async def run():
        await asyncio.gather(
            service.system_one(
                state="one", images=None, questions=NOUL, score_readout=None, set_size=None
            ),
            service.system_one(
                state="two", images=None, questions=NOUL, score_readout=None, set_size=None
            ),
            service.compare(
                a={"text": "alpha"},
                b={"text": "beta"},
                criteria="Which reads better?",
                state=None,
                score_readout=None,
                set_size=None,
            ),
            service.run_preset(
                "sample", state="three", images=None, score_readout=None, set_size=None
            ),
        )

    asyncio.run(run())
    assert backend.max_in_flight == 1
    kinds = [call["kind"] for call in backend.calls]
    assert kinds.count("one") == 3
    assert kinds.count("batch") == 1
