# Benchmarking

Every response includes `timing`. `timing.latency_ms` is server time for that
HTTP call, including image loads. `timing.inference_ms` is the model call
only. `usage.output_tokens` is 0.

For a repeatable record, point `bench` at a running server. Warmup calls are
excluded. `mlx-d1 bench` does not load or download weights.

```bash
mlx-d1 bench --requests 20 --warmup 3
mlx-d1 bench --requests 20 --warmup 3 --payload request.json --json
```

Compare the same body with llama-server. The default body is one `noul`
question, which both servers accept. Extraction, estimates, and boxes are not
in llama.cpp yet, so a payload that uses them will fail on that side.

```bash
llama-server -hf LiquidAI/d1-2B-GGUF:Q8_0 -b 8192 -ub 8192
mlx-d1 bench --requests 20 --warmup 3 --llama-url http://127.0.0.1:8080
```

The report prints count, mean, p50, p95, min, and max in milliseconds, using
client-side time around the POST. Record those numbers on the Mac. This
repository does not ship measured device numbers.
