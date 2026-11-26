import os
from pathlib import Path

import pytest

from mlx_d1.config import load_settings, resolve_config_path
from mlx_d1.errors import ConfigError

ROOT = Path(__file__).resolve().parents[1]


def test_example_config_has_only_the_sample_preset():
    settings = load_settings(config_path=ROOT / "examples" / "config.toml", environ={})
    assert settings.model == "LiquidAI/d1-2B-MLX-8bit"
    assert settings.port == 8091
    assert list(settings.presets) == ["image_artifacts"]
    questions = settings.presets["image_artifacts"].questions
    assert set(questions) == {"extra_limbs", "warped_text", "warped_geometry", "severity"}
    assert questions["severity"]["criteria"] == ["none", "minor", "noticeable", "severe"]


def test_env_overrides_file_and_explicit_overrides_win(tmp_path: Path):
    config = tmp_path / "mlx-d1.toml"
    config.write_text(
        'model = "from-file"\nport = 1111\nallowed_image_dirs = ["/tmp/from-file"]\n',
        encoding="utf-8",
    )
    settings = load_settings(
        config_path=config,
        environ={
            "MLX_D1_MODEL": "from-env",
            "MLX_D1_PORT": "2222",
            "MLX_D1_ALLOWED_IMAGE_DIRS": "",
            "MLX_D1_BACKEND": "fake",
        },
        overrides={"port": 3333},
    )
    assert settings.model == "from-env"
    assert settings.port == 3333
    assert settings.backend == "fake"
    assert settings.allowed_image_dirs == ()


def test_allowed_dirs_use_the_platform_path_separator(tmp_path: Path):
    first = tmp_path / "one"
    second = tmp_path / "two"
    settings = load_settings(
        environ={"MLX_D1_ALLOWED_IMAGE_DIRS": os.pathsep.join((str(first), str(second)))},
    )
    assert settings.allowed_image_dirs == (first, second)


def test_unknown_config_and_preset_fields_fail(tmp_path: Path):
    config = tmp_path / "bad.toml"
    config.write_text(
        """
threshold = 0.8

[presets.sample]
description = "example"
cutoff = 0.5

[presets.sample.questions.q]
type = "noul"
instructions = "Yes?"
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unsupported config fields"):
        load_settings(config_path=config, environ={})


def test_preset_rejects_fields_other_than_questions(tmp_path: Path):
    config = tmp_path / "bad.toml"
    config.write_text(
        """
[presets.sample]
description = "example"
cutoff = 0.5

[presets.sample.questions.q]
type = "noul"
instructions = "Yes?"
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unsupported fields"):
        load_settings(config_path=config, environ={})


def test_allow_download_is_off_unless_opted_in(tmp_path: Path):
    config = tmp_path / "mlx-d1.toml"
    config.write_text("allow_download = true\n", encoding="utf-8")
    assert load_settings(config_path=config, environ={}).allow_download is True
    assert (
        load_settings(config_path=config, environ={"MLX_D1_ALLOW_DOWNLOAD": "0"}).allow_download
        is False
    )
    assert load_settings(environ={}, overrides={"allow_download": True}).allow_download is True
    assert load_settings(environ={}).allow_download is False


def test_resolve_config_path_finds_local_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_config_path(None, environ={}) is None
    path = tmp_path / "mlx-d1.toml"
    path.write_text("port = 1\n", encoding="utf-8")
    assert resolve_config_path(None, environ={}) == Path("mlx-d1.toml")
    assert resolve_config_path(None, environ={"MLX_D1_CONFIG": "other.toml"}) == Path("other.toml")
