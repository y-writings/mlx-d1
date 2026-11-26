import struct
import zlib
from pathlib import Path

import pytest
from PIL import Image

from mlx_d1.errors import ImageError
from mlx_d1.images import load_image_inputs, open_image
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


def _png_header(width: int, height: int) -> bytes:
    """A PNG whose header claims ``width`` by ``height`` and has no pixel data."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", checksum)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")


def test_pixel_cap_rejects_before_the_pillow_error_threshold():
    # Pillow warns, and still decodes, until the image is twice MAX_IMAGE_PIXELS.
    capped = _png_header(width=20, height=8)  # 160 pixels, between 100 and 200
    with pytest.warns(Image.DecompressionBombWarning):
        with pytest.raises(ImageError, match="max_image_pixels"):
            open_image(capped, label="image 1", max_bytes=1024 * 1024, max_pixels=100)


def test_pixel_cap_still_rejects_a_decompression_bomb():
    bomb = _png_header(width=500, height=500)  # 250_000 > 2 * 100
    with pytest.raises(ImageError, match="max_image_pixels"):
        open_image(bomb, label="image 1", max_bytes=1024 * 1024, max_pixels=100)


def test_oversize_data_url():
    with pytest.raises(ImageError, match="max_image_bytes"):
        load_image_inputs([DATA_URL], allowed_dirs=[], max_bytes=8, max_pixels=1_000_000)


def test_pillow_can_open_the_fixture_bytes():
    from io import BytesIO

    with Image.open(BytesIO(PNG_BYTES)) as image:
        assert image.format == "PNG"
