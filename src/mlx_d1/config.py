"""Load settings from defaults, a TOML file, the environment, then explicit overrides."""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlx_d1.errors import ConfigError, QuestionError
from mlx_d1.questions import validate_questions

_PRESET_NAME = re.compile(r"[A-Za-z0-9_-]+")
_SCALAR_KEYS = (
    "model",
    "host",
    "port",
    "backend",
    "score_readout",
    "set_size",
    "max_image_bytes",
    "max_images",
    "max_questions",
    "max_state_bytes",
    "max_image_pixels",
    "max_batch",
    "allowed_image_dirs",
)
_INT_ENV = {
    "MLX_D1_PORT": "port",
    "MLX_D1_MAX_IMAGE_BYTES": "max_image_bytes",
    "MLX_D1_MAX_IMAGES": "max_images",
    "MLX_D1_MAX_QUESTIONS": "max_questions",
    "MLX_D1_MAX_STATE_BYTES": "max_state_bytes",
    "MLX_D1_MAX_IMAGE_PIXELS": "max_image_pixels",
    "MLX_D1_MAX_BATCH": "max_batch",
}


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    questions: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class Settings:
    model: str = "LiquidAI/d1-2B-MLX-8bit"
    host: str = "127.0.0.1"
    port: int = 8091
    backend: str = "mlx"
    allowed_image_dirs: tuple[Path, ...] = ()
    score_readout: str = "median"
    set_size: int | None = None
    max_image_bytes: int = 32 * 1024 * 1024
    max_images: int = 8
    max_questions: int = 128
    max_state_bytes: int = 8 * 1024 * 1024
    max_image_pixels: int = 40_000_000
    max_batch: int = 32
    presets: dict[str, Preset] = field(default_factory=dict)


def load_settings(
    *,
    config_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Settings:
    """Apply defaults, then the TOML file, then environment variables, then overrides."""
    values = _defaults()
    if config_path is not None:
        _apply_file(values, _read_config_file(config_path))
    _apply_env(values, os.environ if environ is None else environ)
    if overrides:
        for key, value in overrides.items():
            if key not in _SCALAR_KEYS:
                raise ConfigError(f"cannot override {key}")
            values[key] = value
    return _finalize(values)


def resolve_config_path(
    explicit: Path | None, environ: Mapping[str, str] | None = None
) -> Path | None:
    """Choose ``--config``, then ``MLX_D1_CONFIG``, then ``./mlx-d1.toml`` when it exists."""
    env = os.environ if environ is None else environ
    if explicit is not None:
        return explicit
    configured = env.get("MLX_D1_CONFIG")
    if configured:
        return Path(configured)
    candidate = Path("mlx-d1.toml")
    if candidate.is_file():
        return candidate
    return None


def _defaults() -> dict[str, Any]:
    return {
        "model": "LiquidAI/d1-2B-MLX-8bit",
        "host": "127.0.0.1",
        "port": 8091,
        "backend": "mlx",
        "allowed_image_dirs": [],
        "score_readout": "median",
        "set_size": None,
        "max_image_bytes": 32 * 1024 * 1024,
        "max_images": 8,
        "max_questions": 128,
        "max_state_bytes": 8 * 1024 * 1024,
        "max_image_pixels": 40_000_000,
        "max_batch": 32,
        "presets": {},
    }


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: config must be a table")
    return data


def _apply_file(values: dict[str, Any], data: dict[str, Any]) -> None:
    unknown = set(data) - set(_SCALAR_KEYS) - {"presets"}
    if unknown:
        raise ConfigError(f"unsupported config fields: {', '.join(sorted(unknown))}")
    for key in _SCALAR_KEYS:
        if key in data:
            values[key] = data[key]
    if "presets" in data:
        values["presets"] = _parse_presets(data["presets"])


def _apply_env(values: dict[str, Any], environ: Mapping[str, str]) -> None:
    string_keys = {
        "MLX_D1_MODEL": "model",
        "MLX_D1_HOST": "host",
        "MLX_D1_BACKEND": "backend",
        "MLX_D1_SCORE_READOUT": "score_readout",
    }
    for env_name, key in string_keys.items():
        if env_name in environ:
            values[key] = environ[env_name]
    for env_name, key in _INT_ENV.items():
        if env_name in environ:
            values[key] = _parse_int_env(env_name, environ[env_name])
    if "MLX_D1_SET_SIZE" in environ:
        raw = environ["MLX_D1_SET_SIZE"].strip().lower()
        if raw in {"", "none", "null"}:
            values["set_size"] = None
        else:
            values["set_size"] = _parse_int_env("MLX_D1_SET_SIZE", raw)
    if "MLX_D1_ALLOWED_IMAGE_DIRS" in environ:
        raw = environ["MLX_D1_ALLOWED_IMAGE_DIRS"]
        if raw.strip() == "":
            values["allowed_image_dirs"] = []
        else:
            values["allowed_image_dirs"] = [part for part in raw.split(os.pathsep) if part]


def _parse_presets(raw: Any) -> dict[str, Preset]:
    if not isinstance(raw, dict):
        raise ConfigError("presets must be a table")
    presets: dict[str, Preset] = {}
    for name, body in raw.items():
        if not isinstance(name, str) or _PRESET_NAME.fullmatch(name) is None:
            raise ConfigError(
                f"preset name {name!r} must contain only letters, digits, '_' and '-'"
            )
        if not isinstance(body, dict):
            raise ConfigError(f"preset {name}: must be a table")
        unknown = set(body) - {"description", "questions"}
        if unknown:
            joined = ", ".join(sorted(str(key) for key in unknown))
            raise ConfigError(
                f"preset {name}: unsupported fields: {joined}. "
                "Presets only hold a description and questions."
            )
        description = body.get("description", "")
        if not isinstance(description, str):
            raise ConfigError(f"preset {name}: description must be a string")
        try:
            questions = validate_questions(body.get("questions"))
        except QuestionError as exc:
            raise ConfigError(f"preset {name}: {exc}") from exc
        presets[name] = Preset(name=name, description=description.strip(), questions=questions)
    return presets


def _parse_int_env(name: str, raw: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _finalize(values: dict[str, Any]) -> Settings:
    backend = values["backend"]
    if backend not in {"mlx", "fake"}:
        raise ConfigError("backend must be 'mlx' or 'fake'")
    readout = values["score_readout"]
    if readout not in {"median", "mean", "argmax"}:
        raise ConfigError("score_readout must be median, mean, or argmax")
    port = values["port"]
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ConfigError("port must be an integer from 1 to 65535")
    set_size = values["set_size"]
    if set_size is not None and (
        isinstance(set_size, bool) or not isinstance(set_size, int) or set_size < 1
    ):
        raise ConfigError("set_size must be a positive integer or null")
    for key in (
        "max_image_bytes",
        "max_images",
        "max_questions",
        "max_state_bytes",
        "max_image_pixels",
        "max_batch",
    ):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError(f"{key} must be a positive integer")
    dirs = values["allowed_image_dirs"]
    if not isinstance(dirs, (list, tuple)):
        raise ConfigError("allowed_image_dirs must be an array of paths")
    paths: list[Path] = []
    for item in dirs:
        if not isinstance(item, (str, Path)) or not str(item).strip():
            raise ConfigError("allowed_image_dirs entries must be non-empty paths")
        paths.append(Path(item).expanduser())
    model = values["model"]
    host = values["host"]
    if not isinstance(model, str) or not model.strip():
        raise ConfigError("model must be a non-empty string")
    if not isinstance(host, str) or not host.strip():
        raise ConfigError("host must be a non-empty string")
    presets = values["presets"]
    if not isinstance(presets, dict):
        raise ConfigError("presets must be a table")
    return Settings(
        model=model.strip(),
        host=host.strip(),
        port=port,
        backend=backend,
        allowed_image_dirs=tuple(paths),
        score_readout=readout,
        set_size=set_size,
        max_image_bytes=values["max_image_bytes"],
        max_images=values["max_images"],
        max_questions=values["max_questions"],
        max_state_bytes=values["max_state_bytes"],
        max_image_pixels=values["max_image_pixels"],
        max_batch=values["max_batch"],
        presets=presets,
    )
