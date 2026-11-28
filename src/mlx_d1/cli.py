"""Command line: serve the model, expose MCP, or measure latency."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from mlx_d1.bench import DEFAULT_PAYLOAD, bench_url, format_report
from mlx_d1.config import load_settings, resolve_config_path
from mlx_d1.errors import ConfigError, ModelNotAvailable
from mlx_d1.server import create_app


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = args.func(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except ModelNotAvailable as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    except BrokenPipeError:
        raise SystemExit(0) from None
    raise SystemExit(code)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-d1", description="Resident decision server for LiquidAI d1."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Load the model once and serve HTTP")
    serve.add_argument("--config", type=Path, help="TOML config file")
    serve.add_argument("--model", help="Model id or local path")
    serve.add_argument("--host", help="Bind address (default 127.0.0.1)")
    serve.add_argument("--port", type=int, help="Bind port (default 8091)")
    serve.add_argument(
        "--backend", choices=("mlx", "fake"), help="mlx loads weights; fake answers without them"
    )
    serve.add_argument("--score-readout", choices=("median", "mean", "argmax"))
    serve.add_argument(
        "--allow-dir", action="append", help="Allow image paths under this directory (repeatable)"
    )
    serve.add_argument(
        "--allow-download",
        action="store_true",
        help="Download a missing model without asking (launchd and other non-interactive starts)",
    )
    serve.set_defaults(func=cmd_serve)

    mcp = sub.add_parser("mcp", help="Serve MCP tools over stdio, calling the HTTP server")
    mcp.add_argument(
        "--url", help="Judging server URL (default MLX_D1_URL or http://127.0.0.1:8091)"
    )
    mcp.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds")
    mcp.set_defaults(func=cmd_mcp)

    bench = sub.add_parser("bench", help="Measure per-request latency")
    bench.add_argument("--url", default=None, help="mlx-d1 base URL")
    bench.add_argument(
        "--llama-url", default=None, help="Optional llama-server base URL for the same payload"
    )
    bench.add_argument("--path", default="/v1/systemone", help="Request path on both servers")
    bench.add_argument(
        "--payload", type=Path, help="JSON request file (default is one noul question)"
    )
    bench.add_argument("--requests", type=int, default=20)
    bench.add_argument("--warmup", type=int, default=3)
    bench.add_argument("--timeout", type=float, default=120.0)
    bench.add_argument("--json", action="store_true", help="Print the report as JSON")
    bench.set_defaults(func=cmd_bench)
    return parser


def cmd_serve(args: argparse.Namespace) -> int:
    settings = load_settings(
        config_path=resolve_config_path(args.config), overrides=_serve_overrides(args)
    )
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    log = logging.getLogger("mlx_d1")
    weights_path = None
    if settings.backend == "mlx":
        from mlx_d1.weights import prepare_weights

        try:
            weights_path = prepare_weights(settings.model, allow_download=settings.allow_download)
        except ModelNotAvailable as exc:
            print(str(exc), file=sys.stderr)
            return 1
    log.info(
        "serving http://%s:%s backend=%s model=%s",
        settings.host,
        settings.port,
        settings.backend,
        settings.model,
    )
    import uvicorn

    uvicorn.run(
        create_app(settings, weights_path=weights_path),
        host=settings.host,
        port=settings.port,
        log_level="info",
    )
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from mlx_d1.client import DecisionClient
    from mlx_d1.mcp_server import build_mcp

    url = args.url or os.environ.get("MLX_D1_URL", "http://127.0.0.1:8091")
    client = DecisionClient(url, timeout=args.timeout)
    server = build_mcp(client)
    server.run(transport="stdio")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    payload = DEFAULT_PAYLOAD
    if args.payload is not None:
        payload = json.loads(args.payload.read_text(encoding="utf-8"))
    url = args.url or os.environ.get("MLX_D1_URL", "http://127.0.0.1:8091")
    report: dict[str, Any] = {"path": args.path, "requests": args.requests, "warmup": args.warmup}
    try:
        target_stats = bench_url(
            url,
            args.path,
            payload,
            requests=args.requests,
            warmup=args.warmup,
            timeout=args.timeout,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    report["target"] = {"url": url, **target_stats}
    llama_error: str | None = None
    if args.llama_url:
        try:
            llama_stats = bench_url(
                args.llama_url,
                args.path,
                payload,
                requests=args.requests,
                warmup=args.warmup,
                timeout=args.timeout,
            )
            report["llama"] = {"url": args.llama_url, **llama_stats}
        except Exception as exc:
            llama_error = str(exc)
            report["llama"] = {"url": args.llama_url, "error": llama_error}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report("mlx-d1", url, target_stats))
        if "llama" in report and llama_error is None:
            print(format_report("llama-server", args.llama_url, report["llama"]))
        elif llama_error is not None:
            print(f"llama-server  {args.llama_url}\n  error: {llama_error}", file=sys.stderr)
    return 1 if llama_error else 0


def _serve_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.model:
        overrides["model"] = args.model
    if args.host:
        overrides["host"] = args.host
    if args.port is not None:
        overrides["port"] = args.port
    if args.backend:
        overrides["backend"] = args.backend
    if args.score_readout:
        overrides["score_readout"] = args.score_readout
    if args.allow_dir is not None:
        overrides["allowed_image_dirs"] = args.allow_dir
    if getattr(args, "allow_download", False):
        overrides["allow_download"] = True
    return overrides
