"""Load request images from an allowed path, base64, or a data URL."""

from __future__ import annotations

import base64
import re
import threading
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from mlx_d1.errors import ImageError

_OPEN_LOCK = threading.Lock()
_DATA_URL = re.compile(
    r"^data:(?P<mime>image/[A-Za-z0-9.+-]+)"
    r"(?:;[A-Za-z0-9.+-]+=[A-Za-z0-9.+-]+)*"
    r";base64,(?P<data>[A-Za-z0-9+/=\s]+)$",
    re.IGNORECASE | re.DOTALL,
)
_BASE64_TEXT = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


def load_image_inputs(
    specs: Sequence[str],
    *,
    allowed_dirs: Sequence[Path],
    max_bytes: int,
    max_pixels: int,
) -> list[Image.Image]:
    """Decode each spec into a PIL image.

    A value is a file path unless it is an ``image/*`` data URL or a long
    base64 string that decodes to image bytes. Paths are read only when their
    resolved location is a file inside one of ``allowed_dirs``.
    """
    images: list[Image.Image] = []
    for index, spec in enumerate(specs, start=1):
        label = f"image {index}"
        if not isinstance(spec, str) or not spec.strip():
            raise ImageError(f"{label} is empty")
        kind, raw = _decode_spec(spec, allowed_dirs=allowed_dirs, max_bytes=max_bytes, label=label)
        images.append(
            open_image(raw, label=f"{label} ({kind})", max_bytes=max_bytes, max_pixels=max_pixels)
        )
    return images


def open_image(raw: bytes, *, label: str, max_bytes: int, max_pixels: int) -> Image.Image:
    """Open image bytes, enforcing size and pixel limits."""
    if len(raw) > max_bytes:
        raise ImageError(f"{label} exceeds max_image_bytes ({max_bytes})")
    with _OPEN_LOCK:
        previous = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = max_pixels
        try:
            with Image.open(BytesIO(raw)) as image:
                image.load()
                return image.copy()
        except Image.DecompressionBombError as exc:
            raise ImageError(f"{label} exceeds max_image_pixels ({max_pixels})") from exc
        except (UnidentifiedImageError, OSError) as exc:
            raise ImageError(f"{label} is not a readable image") from exc
        finally:
            Image.MAX_IMAGE_PIXELS = previous


def _decode_spec(
    spec: str,
    *,
    allowed_dirs: Sequence[Path],
    max_bytes: int,
    label: str,
) -> tuple[str, bytes]:
    if spec.startswith("data:"):
        return "data-url", _decode_data_url(spec, max_bytes=max_bytes, label=label)
    if _is_base64_candidate(spec):
        # A real file with this name wins. Long image payloads contain '/', so
        # a leading slash is not enough to call something a path.
        if _exists_as_file(spec):
            return "path", _read_allowed_file(spec, allowed_dirs, max_bytes=max_bytes, label=label)
        raw = _decode_base64(spec, max_bytes=max_bytes, label=label)
        if _looks_like_image(raw):
            return "base64", raw
        raise ImageError(f"{label} base64 is not an image")
    return "path", _read_allowed_file(spec, allowed_dirs, max_bytes=max_bytes, label=label)


def _decode_data_url(spec: str, *, max_bytes: int, label: str) -> bytes:
    match = _DATA_URL.fullmatch(spec.strip())
    if match is None:
        raise ImageError(f"{label} data URL must be image/*;base64,...")
    return _decode_base64(match.group("data"), max_bytes=max_bytes, label=label)


def _decode_base64(spec: str, *, max_bytes: int, label: str) -> bytes:
    compact = "".join(spec.split())
    if not _BASE64_TEXT.fullmatch(compact):
        raise ImageError(f"{label} is not valid base64")
    if len(compact) * 3 // 4 > max_bytes:
        raise ImageError(f"{label} exceeds max_image_bytes ({max_bytes})")
    try:
        return base64.b64decode(compact, validate=True)
    except ValueError as exc:
        raise ImageError(f"{label} is not valid base64") from exc


def _is_base64_candidate(spec: str) -> bool:
    compact = "".join(spec.split())
    return len(compact) >= 64 and _BASE64_TEXT.fullmatch(compact) is not None


def _looks_like_image(raw: bytes) -> bool:
    return raw.startswith((b"\x89PNG", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"BM", b"RIFF"))


def _exists_as_file(raw: str) -> bool:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        return candidate.resolve(strict=True).is_file()
    except OSError:
        return False


def _read_allowed_file(
    raw: str, allowed_dirs: Sequence[Path], *, max_bytes: int, label: str
) -> bytes:
    path = _resolve_allowed_path(raw, allowed_dirs)
    size = path.stat().st_size
    if size > max_bytes:
        raise ImageError(f"{label} exceeds max_image_bytes ({max_bytes})")
    return path.read_bytes()


def _resolve_allowed_path(raw: str, allowed_dirs: Sequence[Path]) -> Path:
    if "\x00" in raw:
        raise ImageError("image path contains NUL")
    if not allowed_dirs:
        raise ImageError("image file paths are disabled until allowed_image_dirs is configured")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        real = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ImageError(f"image file does not exist: {raw}") from exc
    if not real.is_file():
        raise ImageError(f"image path is not a file: {raw}")
    roots: list[Path] = []
    for root in allowed_dirs:
        try:
            root_real = Path(root).expanduser().resolve(strict=True)
        except FileNotFoundError:
            continue
        if root_real.is_dir():
            roots.append(root_real)
    if not roots:
        raise ImageError("no allowed image directory is available")
    if any(root in real.parents for root in roots):
        return real
    raise ImageError(f"image path is outside the allowed directories: {raw}")
