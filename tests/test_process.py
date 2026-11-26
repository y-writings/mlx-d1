import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_server_process_answers_repeated_requests(tmp_path: Path):
    port = _free_port()
    config = tmp_path / "mlx-d1.toml"
    config.write_text(
        f'backend = "fake"\nhost = "127.0.0.1"\nport = {port}\nmodel = "process-fake"\n',
        encoding="utf-8",
    )
    log_path = tmp_path / "server.log"
    env = {key: value for key, value in os.environ.items() if not key.startswith("MLX_D1_")}
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, "-m", "mlx_d1", "serve", "--config", str(config), "--backend", "fake"],
            cwd=tmp_path,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            health = _wait_health(f"http://127.0.0.1:{port}", process, log_path)
            assert health["status"] == "ok"
            assert health["backend"] == "fake"
            payload = {"state": "ping", "questions": {"q": {"type": "noul", "instructions": "Up?"}}}
            with httpx.Client(timeout=5) as http:
                first = http.post(f"http://127.0.0.1:{port}/v1/systemone", json=payload)
                second = http.post(f"http://127.0.0.1:{port}/v1/systemone", json=payload)
            assert first.status_code == second.status_code == 200
            assert first.json()["answers"]["q"]["type"] == "noul"
            assert second.json()["model"] == "process-fake"
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _wait_health(url: str, process: subprocess.Popen, log_path: Path) -> dict:
    deadline = time.time() + 15
    last_error = ""
    while time.time() < deadline:
        if process.poll() is not None:
            raise AssertionError(log_path.read_text(encoding="utf-8"))
        try:
            response = httpx.get(url + "/health", timeout=0.5)
        except httpx.HTTPError as exc:
            last_error = str(exc)
        else:
            if response.status_code == 200:
                body = response.json()
                assert isinstance(body, dict)
                return body
            last_error = response.text
        time.sleep(0.1)
    raise AssertionError(f"{last_error}\n{log_path.read_text(encoding='utf-8')}")


def test_bench_summary_shape_is_stable():
    # Imported here so a bench-only failure points at the helper, not the process.
    from mlx_d1.bench import summarize

    stats = summarize([1.0, 3.0])
    assert json.dumps(stats)
