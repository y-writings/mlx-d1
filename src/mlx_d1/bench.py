"""Measure per-request latency against this server and, optionally, llama-server."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import httpx

from mlx_d1.client import DecisionClientError, error_message

DEFAULT_PAYLOAD: dict[str, Any] = {
    "state": "I was charged twice. Please refund one charge.",
    "questions": {
        "refund": {
            "type": "noul",
            "instructions": "Is the customer asking for a refund?",
        }
    },
}


def percentile(samples: list[float], fraction: float) -> float:
    """Linear percentile. ``fraction`` is between 0 and 1."""
    if not samples:
        raise ValueError("no samples")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = int(position) if position == int(position) else int(position) + 1
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def summarize(samples_ms: list[float]) -> dict[str, float | int]:
    if not samples_ms:
        raise ValueError("no samples")
    return {
        "count": len(samples_ms),
        "mean_ms": round(statistics.fmean(samples_ms), 3),
        "min_ms": round(min(samples_ms), 3),
        "max_ms": round(max(samples_ms), 3),
        "p50_ms": round(percentile(samples_ms, 0.50), 3),
        "p95_ms": round(percentile(samples_ms, 0.95), 3),
    }


def measure(
    post: Callable[[dict[str, Any]], Any],
    payload: dict[str, Any],
    *,
    requests: int,
    warmup: int,
) -> dict[str, float | int]:
    """Time ``requests`` posts. ``warmup`` posts run first and are excluded."""
    if requests < 1:
        raise ValueError("requests must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must be at least 0")
    for _ in range(warmup):
        post(payload)
    samples: list[float] = []
    for _ in range(requests):
        started = time.perf_counter()
        post(payload)
        samples.append((time.perf_counter() - started) * 1000)
    return summarize(samples)


def bench_url(
    url: str,
    path: str,
    payload: dict[str, Any],
    *,
    requests: int,
    warmup: int,
    timeout: float,
) -> dict[str, float | int]:
    target = url.rstrip("/") + path
    with httpx.Client(timeout=timeout) as client:

        def post(body: dict[str, Any]) -> Any:
            response = client.post(target, json=body)
            if response.status_code >= 400:
                raise DecisionClientError(response.status_code, error_message(response))
            return response

        return measure(post, payload, requests=requests, warmup=warmup)


def format_report(name: str, url: str, stats: dict[str, Any]) -> str:
    return (
        f"{name}  {url}\n"
        f"  count {stats['count']}  mean {stats['mean_ms']:.1f} ms  "
        f"p50 {stats['p50_ms']:.1f}  p95 {stats['p95_ms']:.1f}  "
        f"min {stats['min_ms']:.1f}  max {stats['max_ms']:.1f}"
    )
