"""Construct the configured backend."""

from __future__ import annotations

from mlx_d1.backends.base import DecisionBackend
from mlx_d1.config import Settings
from mlx_d1.errors import ConfigError


def load_backend(settings: Settings) -> DecisionBackend:
    if settings.backend == "fake":
        from mlx_d1.backends.fake import FakeBackend

        return FakeBackend(model_id=settings.model)
    if settings.backend == "mlx":
        from mlx_d1.backends.mlx import MlxVlmBackend

        return MlxVlmBackend(settings.model)
    raise ConfigError("backend must be 'mlx' or 'fake'")
