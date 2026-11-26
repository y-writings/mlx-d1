from pathlib import Path

import pytest
from PIL import Image

from mlx_d1.errors import ImageError
from mlx_d1.images import load_image_inputs
from tests.conftest import DATA_URL, PNG_BASE64, PNG_BYTES


def _load(specs, allowed):
    return load_image_inputs(
        specs, allowed_dirs=allowed, max_bytes=1024 * 1024, max_pixels=1_000_000
    )


def test_data_url_and_raw_base64(tmp_path: Path):
    images = _load([DATA_URL, PNG_BASE64], [tmp_path])
    assert len(images) == 2
    assert all(image.size == (1, 1) for image in images)


def test_allowed_path_relative_path_and_sibling_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    allowed = tmp_path / "allowed"
    sibling = tmp_path / "allowed-extra"
    allowed.mkdir()
    sibling.mkdir()
    inside = allowed / "pixel.png"
    outside = sibling / "pixel.png"
    inside.write_bytes(PNG_BYTES)
    outside.write_bytes(PNG_BYTES)
    monkeypatch.chdir(allowed)
    loaded = _load(["pixel.png", str(inside)], [allowed])
    assert len(loaded) == 2
    with pytest.raises(ImageError, match="outside"):
        _load([str(outside)], [allowed])


def test_symlink_escape_is_rejected(tmp_path: Path):
    allowed = tmp_path / "allowed"
    secret = tmp_path / "secret"
    allowed.mkdir()
    secret.mkdir()
    hidden = secret / "pixel.png"
    hidden.write_bytes(PNG_BYTES)
    link = allowed / "link.png"
    link.symlink_to(hidden)
    with pytest.raises(ImageError, match="outside"):
        _load([str(link)], [allowed])


def test_symlink_inside_the_directory_is_allowed(tmp_path: Path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    target = allowed / "pixel.png"
    target.write_bytes(PNG_BYTES)
    link = allowed / "link.png"
    link.symlink_to(target)
    assert _load([str(link)], [allowed])[0].size == (1, 1)


def test_missing_allowlist_rejects_paths_but_not_data_urls(tmp_path: Path):
    with pytest.raises(ImageError, match="allowed_image_dirs"):
        _load([str(tmp_path / "pixel.png")], [])
    assert isinstance(_load([DATA_URL], []), list)


def test_directory_and_nul_are_rejected(tmp_path: Path):
    with pytest.raises(ImageError, match="not a file"):
        _load([str(tmp_path)], [tmp_path])
    with pytest.raises(ImageError, match="NUL"):
        _load(["bad\x00.png"], [tmp_path])


def test_oversize_data_url():
    with pytest.raises(ImageError, match="max_image_bytes"):
        load_image_inputs([DATA_URL], allowed_dirs=[], max_bytes=8, max_pixels=1_000_000)


def test_pillow_can_open_the_fixture_bytes():
    from io import BytesIO

    with Image.open(BytesIO(PNG_BYTES)) as image:
        assert image.format == "PNG"
