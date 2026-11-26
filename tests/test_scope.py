from pathlib import Path


def test_server_code_is_not_tied_to_one_product():
    root = Path(__file__).resolve().parents[1] / "src" / "mlx_d1"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py")).lower()
    for word in ("comfyui", "comfy"):
        assert word not in text
