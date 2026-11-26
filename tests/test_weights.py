from pathlib import Path

import pytest

from mlx_d1.errors import ModelNotAvailable
from mlx_d1.weights import prepare_weights, snapshot_fetch, snapshot_locate


def _checkpoint(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")
    return path


def test_local_directory_does_not_download_or_prompt(tmp_path: Path):
    folder = _checkpoint(tmp_path / "d1")
    calls = {"locate": 0, "fetch": 0, "ask": 0}

    def locate(_repo: str) -> Path | None:
        calls["locate"] += 1
        return None

    def fetch(_repo: str) -> Path:
        calls["fetch"] += 1
        raise AssertionError("download")

    def ask() -> str:
        calls["ask"] += 1
        return "yes"

    found = prepare_weights(
        str(folder),
        allow_download=True,
        interactive=True,
        input_fn=ask,
        locate=locate,
        fetch=fetch,
        describe_size=lambda _repo: "about 2.22 GB",
        echo=lambda _message: None,
    )
    assert found == folder.resolve()
    assert calls == {"locate": 0, "fetch": 0, "ask": 0}


def test_cached_repo_does_not_prompt(tmp_path: Path):
    cached = _checkpoint(tmp_path / "cache")
    fetched = {"n": 0}

    found = prepare_weights(
        "LiquidAI/d1-2B-MLX-8bit",
        allow_download=False,
        interactive=True,
        input_fn=lambda: (_ for _ in ()).throw(AssertionError("prompt")),
        locate=lambda _repo: cached,
        fetch=lambda _repo: fetched.__setitem__("n", 1),
        describe_size=lambda _repo: None,
        echo=lambda _message: None,
    )
    assert found == cached.resolve()
    assert fetched["n"] == 0


def test_yes_downloads_and_no_explains_how_to_point_at_a_folder(tmp_path: Path):
    downloaded = _checkpoint(tmp_path / "downloaded")
    seen: list[str] = []

    def fetch(repo: str) -> Path:
        assert repo == "LiquidAI/d1-2B-MLX-8bit"
        return downloaded

    found = prepare_weights(
        "LiquidAI/d1-2B-MLX-8bit",
        allow_download=False,
        interactive=True,
        input_fn=lambda: "yes",
        locate=lambda _repo: None,
        fetch=fetch,
        describe_size=lambda _repo: "about 2.22 GB",
        echo=seen.append,
    )
    assert found == downloaded.resolve()
    assert any("not available locally" in line and "2.22 GB" in line for line in seen)
    assert any("Download it now?" in line for line in seen)

    with pytest.raises(ModelNotAvailable) as declined:
        prepare_weights(
            "LiquidAI/d1-2B-MLX-8bit",
            allow_download=False,
            interactive=True,
            input_fn=lambda: "n",
            locate=lambda _repo: None,
            fetch=lambda _repo: (_ for _ in ()).throw(AssertionError("download")),
            describe_size=lambda _repo: "about 2.22 GB",
            echo=lambda _message: None,
        )
    message = str(declined.value)
    assert "huggingface-cli download LiquidAI/d1-2B-MLX-8bit" in message
    assert "model in mlx-d1.toml" in message
    assert "MLX_D1_MODEL" in message
    assert "--model" in message
    assert message.index("mlx-d1.toml") < message.index("MLX_D1_MODEL") < message.index("--model")


def test_no_tty_never_downloads_unless_allowed(tmp_path: Path):
    downloaded = _checkpoint(tmp_path / "auto")

    with pytest.raises(ModelNotAvailable) as refused:
        prepare_weights(
            "LiquidAI/d1-2B-MLX-8bit",
            allow_download=False,
            interactive=False,
            input_fn=lambda: "yes",
            locate=lambda _repo: None,
            fetch=lambda _repo: (_ for _ in ()).throw(AssertionError("download")),
            describe_size=lambda _repo: "about 2.22 GB",
            echo=lambda _message: None,
        )
    text = str(refused.value)
    assert "No terminal is attached" in text
    assert "MLX_D1_ALLOW_DOWNLOAD" in text
    assert "--allow-download" in text

    found = prepare_weights(
        "LiquidAI/d1-2B-MLX-8bit",
        allow_download=True,
        interactive=False,
        input_fn=lambda: (_ for _ in ()).throw(AssertionError("prompt")),
        locate=lambda _repo: None,
        fetch=lambda _repo: downloaded,
        describe_size=lambda _repo: None,
        echo=lambda _message: None,
    )
    assert found == downloaded.resolve()


def test_incomplete_cache_can_still_be_downloaded(tmp_path: Path):
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "config.json").write_text("{}", encoding="utf-8")
    downloaded = _checkpoint(tmp_path / "full")
    fetched = {"n": 0}

    def fetch(_repo: str) -> Path:
        fetched["n"] += 1
        return downloaded

    found = prepare_weights(
        "LiquidAI/d1-2B-MLX-8bit",
        allow_download=True,
        interactive=False,
        input_fn=lambda: (_ for _ in ()).throw(AssertionError("prompt")),
        locate=lambda _repo: partial,
        fetch=fetch,
        describe_size=lambda _repo: None,
        echo=lambda _message: None,
    )
    assert found == downloaded.resolve()
    assert fetched["n"] == 1

    with pytest.raises(ModelNotAvailable, match="No terminal is attached"):
        prepare_weights(
            "LiquidAI/d1-2B-MLX-8bit",
            allow_download=False,
            interactive=False,
            locate=lambda _repo: partial,
            fetch=lambda _repo: (_ for _ in ()).throw(AssertionError("download")),
            describe_size=lambda _repo: None,
            echo=lambda _message: None,
        )


def test_offline_hub_blocks_a_client_imported_earlier():
    import httpx
    import huggingface_hub.constants as constants
    from huggingface_hub.errors import OfflineModeIsEnabled
    from huggingface_hub.utils._http import hf_request_event_hook

    from mlx_d1.weights import offline_hub

    constants.HF_HUB_OFFLINE = False
    request = httpx.Request("GET", "https://huggingface.co/api/models/LiquidAI/d1-2B-MLX-8bit")
    hf_request_event_hook(request)
    with offline_hub():
        assert constants.is_offline_mode() is True
        with pytest.raises(OfflineModeIsEnabled):
            hf_request_event_hook(request)
    assert constants.HF_HUB_OFFLINE is False
    hf_request_event_hook(request)


def test_missing_path_is_not_treated_as_a_repo(tmp_path: Path):
    missing = tmp_path / "missing-model"
    with pytest.raises(ModelNotAvailable, match="does not exist"):
        prepare_weights(
            str(missing),
            locate=lambda _repo: (_ for _ in ()).throw(AssertionError("locate")),
            fetch=lambda _repo: (_ for _ in ()).throw(AssertionError("download")),
            describe_size=lambda _repo: None,
            echo=lambda _message: None,
        )


def test_snapshot_lookup_is_local_only(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []

    def snapshot_download(repo_id: str, **kwargs):
        calls.append({"repo_id": repo_id, **kwargs})
        if kwargs.get("local_files_only"):
            from huggingface_hub.errors import LocalEntryNotFoundError

            raise LocalEntryNotFoundError("not cached")
        raise AssertionError("network download")

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    assert snapshot_locate("LiquidAI/d1-2B-MLX-8bit") is None
    assert calls[0]["local_files_only"] is True
    assert calls[0]["repo_id"] == "LiquidAI/d1-2B-MLX-8bit"


def test_snapshot_fetch_is_the_explicit_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: list[dict] = []

    def snapshot_download(repo_id: str, **kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    assert snapshot_fetch("org/model") == tmp_path
    assert calls[0].get("local_files_only") is not True


def test_serve_exits_when_the_model_is_declined(monkeypatch: pytest.MonkeyPatch):
    from mlx_d1.cli import main

    def refuse(*_args, **_kwargs):
        raise ModelNotAvailable("not downloaded\nmodel in mlx-d1.toml\nMLX_D1_MODEL\n--model")

    monkeypatch.setattr("mlx_d1.weights.prepare_weights", refuse)
    with pytest.raises(SystemExit) as exited:
        main(["serve", "--backend", "mlx", "--model", "LiquidAI/d1-2B-MLX-8bit"])
    assert exited.value.code == 1
