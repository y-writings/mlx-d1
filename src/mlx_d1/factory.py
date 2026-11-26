"""Construct the configured backend."""

from __future__ import annotations

from pathlib import Path

from mlx_d1.backends.base import DecisionBackend
from mlx_d1.config import Settings
from mlx_d1.errors import ConfigError


def load_backend(settings: Settings, *, weights_path: Path | None = None) -> DecisionBackend:
    if settings.backend == "fake":
        from mlx_d1.backends.fake import FakeBackend

        return FakeBackend(model_id=settings.model)
    if settings.backend == "mlx":
        from mlx_d1.backends.mlx import MlxVlmBackend
        from mlx_d1.weights import prepare_weights

        path = weights_path or prepare_weights(
            settings.model, allow_download=settings.allow_download
        )
        return MlxVlmBackend(settings.model, weights_path=path)
    raise ConfigError("backend must be 'mlx' or 'fake'")
