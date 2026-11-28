"""Find d1 weights on disk, or download them only after an explicit yes.

``mlx_vlm.load`` downloads a Hugging Face repo id whenever the path is missing,
and it does not take a local-only flag. This module therefore never hands a
repo id to the loader. It returns a directory that already exists. A download
is a separate call, used only after the user agrees or ``allow_download`` is set.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mlx_d1.errors import ModelNotAvailable

# Same patterns mlx_vlm.utils.get_model_path passes to snapshot_download.
SNAPSHOT_PATTERNS = (
    "*.json",
    "*.jsonl",
    "*.safetensors",
    "*.py",
    "*.model",
    "*.tiktoken",
    "*.txt",
    "*.jinja",
)
# Used when the Hub metadata call cannot report a size. The figure is the
# model card's weight size, not a measured download from this process.
_KNOWN_SIZES = {"LiquidAI/d1-2B-MLX-8bit": "about 2.22 GB"}
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_YES = {"y", "yes"}

Locate = Callable[[str], Path | None]
Fetch = Callable[[str], Path]
Echo = Callable[[str], None]


def prepare_weights(
    spec: str,
    *,
    allow_download: bool = False,
    interactive: bool | None = None,
    input_fn: Callable[[], str] | None = None,
    locate: Locate | None = None,
    fetch: Fetch | None = None,
    describe_size: Callable[[str], str | None] | None = None,
    echo: Echo | None = None,
) -> Path:
    """Return a local checkpoint directory. Download only when that is allowed.

    A local directory, or a repo id that is already complete in the Hugging
    Face cache, is returned with no prompt and no download. Anything else asks
    on a terminal, unless ``allow_download`` is set. Without a terminal and
    without that opt-in, this raises ``ModelNotAvailable`` and does not fetch.
    """
    locate = locate or snapshot_locate
    fetch = fetch or snapshot_fetch
    describe_size = describe_size or hub_size
    echo = echo or _echo
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stderr.isatty()
    if input_fn is None:
        input_fn = input

    local = _existing_directory(spec)
    if local is not None:
        return require_checkpoint(local)
    if not _is_repo_id(spec):
        raise ModelNotAvailable(_missing_path_message(spec))

    cached = locate(spec)
    if cached is not None and is_complete_checkpoint(Path(cached)):
        return require_checkpoint(Path(cached))

    if allow_download:
        size = describe_size(spec)
        detail = f" ({size})" if size else ""
        echo(f"Downloading {spec}{detail}.")
        return require_checkpoint(Path(fetch(spec)))

    size = describe_size(spec)
    if not interactive:
        raise ModelNotAvailable(_unattended_message(spec, size))
    if _confirm(spec, size, input_fn=input_fn, echo=echo):
        echo(f"Downloading {spec}.")
        return require_checkpoint(Path(fetch(spec)))
    raise ModelNotAvailable(_declined_message(spec))


def snapshot_locate(repo_id: str) -> Path | None:
    """Return the cached snapshot, or None. This call sets ``local_files_only``."""
    snapshot_download, missing = _hub()
    try:
        found = snapshot_download(
            repo_id,
            local_files_only=True,
            allow_patterns=list(SNAPSHOT_PATTERNS),
        )
    except missing:
        return None
    return Path(found)


def snapshot_fetch(repo_id: str) -> Path:
    """Download a snapshot. Call this only after the user opted in."""
    snapshot_download, _missing = _hub()
    found = snapshot_download(repo_id, allow_patterns=list(SNAPSHOT_PATTERNS))
    return Path(found)


def hub_size(repo_id: str) -> str | None:
    """Best-effort size label. A failed metadata lookup is not a weight download."""
    try:
        from huggingface_hub import model_info
    except ImportError:
        return _KNOWN_SIZES.get(repo_id)
    try:
        info = model_info(repo_id, files_metadata=True)
    except Exception:
        return _KNOWN_SIZES.get(repo_id)
    total = 0
    for sibling in info.siblings or []:
        size = getattr(sibling, "size", None)
        if isinstance(size, int) and size > 0:
            total += size
    if total <= 0:
        return _KNOWN_SIZES.get(repo_id)
    return _format_bytes(total)


def offline_hub():
    """Context manager: refuse Hub network access for the duration of ``load``."""
    return _OfflineHub()


class _OfflineHub:
    """Block Hub HTTP calls, including clients already imported.

    ``huggingface_hub`` copies ``HF_HUB_OFFLINE`` into ``constants`` at import
    time, and the HTTP layer reads that value rather than the environment.
    Setting the environment alone does not stop a later ``load`` from fetching.
    """

    def __enter__(self) -> None:
        self._previous_env = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        self._previous_flags: list[tuple[Any, Any]] = []
        for name, module in list(sys.modules.items()):
            if module is None or not name.startswith("huggingface_hub"):
                continue
            values = getattr(module, "__dict__", None)
            if not values or "HF_HUB_OFFLINE" not in values:
                continue
            self._previous_flags.append((module, values["HF_HUB_OFFLINE"]))
            module.HF_HUB_OFFLINE = True

    def __exit__(self, *_exc: object) -> None:
        for module, previous in reversed(self._previous_flags):
            module.HF_HUB_OFFLINE = previous
        if self._previous_env is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = self._previous_env
        # A first import during the context bakes the flag in as True. Reset
        # any copy we did not capture so it matches the restored environment.
        restored = {id(module) for module, _previous in self._previous_flags}
        offline = _env_is_offline()
        for name, module in list(sys.modules.items()):
            if module is None or not name.startswith("huggingface_hub") or id(module) in restored:
                continue
            values = getattr(module, "__dict__", None)
            if values and "HF_HUB_OFFLINE" in values:
                module.HF_HUB_OFFLINE = offline


def _existing_directory(spec: str) -> Path | None:
    path = Path(spec).expanduser()
    if path.is_dir():
        return path.resolve()
    return None


def _is_repo_id(spec: str) -> bool:
    return _REPO_ID.fullmatch(spec) is not None and not Path(spec).exists()


def is_complete_checkpoint(path: Path) -> bool:
    """True when the directory has the files ``mlx_vlm.load`` needs."""
    resolved = path.expanduser()
    return (
        resolved.is_dir()
        and (resolved / "config.json").is_file()
        and any(resolved.glob("*.safetensors"))
    )


def require_checkpoint(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not is_complete_checkpoint(resolved):
        raise ModelNotAvailable(
            f"{resolved} is not a complete MLX checkpoint. "
            "It needs config.json and at least one .safetensors file."
        )
    return resolved


def _confirm(repo_id: str, size: str | None, *, input_fn: Callable[[], str], echo: Echo) -> bool:
    detail = f" ({size})" if size else ""
    echo(f"{repo_id} is not available locally{detail}.")
    echo("Download it now? [y/N] ")
    try:
        answer = input_fn()
    except EOFError:
        answer = ""
    return answer.strip().lower() in _YES


def _declined_message(repo_id: str) -> str:
    folder = repo_id.split("/", 1)[-1]
    return (
        f"{repo_id} was not downloaded.\n\n"
        "Download it yourself, for example:\n\n"
        f"  huggingface-cli download {repo_id} --local-dir ./{folder}\n\n"
        "Then point mlx-d1 at that folder. Later sources replace earlier ones:\n"
        "  1. model in mlx-d1.toml\n"
        "  2. the MLX_D1_MODEL environment variable\n"
        "  3. --model\n\n"
        "Examples:\n"
        f'  model = "/absolute/path/to/{folder}"\n'
        f"  MLX_D1_MODEL=/absolute/path/to/{folder} mlx-d1 serve\n"
        f"  mlx-d1 serve --model /absolute/path/to/{folder}\n"
    )


def _unattended_message(repo_id: str, size: str | None) -> str:
    detail = f" ({size})" if size else ""
    return (
        f"{repo_id} is not available locally{detail}.\n"
        "No terminal is attached, so mlx-d1 will not download it.\n"
        "Opt in with allow_download = true in mlx-d1.toml, "
        "MLX_D1_ALLOW_DOWNLOAD=1, or --allow-download.\n\n" + _declined_message(repo_id)
    )


def _missing_path_message(spec: str) -> str:
    return (
        f"Model path does not exist: {spec}\n"
        "Pass a local checkpoint directory, or a Hugging Face repo id such as "
        "LiquidAI/d1-2B-MLX-8bit.\n"
        "A relative directory must exist, or start with ./ so it is not read as a repo id."
    )


def _format_bytes(num_bytes: int) -> str:
    gib = num_bytes / (1024**3)
    if gib >= 0.1:
        return f"about {gib:.2f} GB"
    return f"about {num_bytes / (1024**2):.0f} MB"


def _env_is_offline() -> bool:
    raw = os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE")
    if raw is None:
        return False
    return raw.upper() in {"1", "ON", "YES", "TRUE"}


def _echo(message: str) -> None:
    print(message, file=sys.stderr)


def _hub() -> tuple[Callable[..., Any], type[BaseException]]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ModelNotAvailable(
            'huggingface_hub is not installed. On Apple Silicon run: pip install -e ".[mlx]"'
        ) from exc
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:
        from huggingface_hub.utils import LocalEntryNotFoundError
    return snapshot_download, LocalEntryNotFoundError
