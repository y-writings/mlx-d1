# Running the server

`mlx-d1 serve` binds `127.0.0.1:8091` unless you change the host or port.
There is no authentication. Do not expose the port on a shared network.

Run one worker. A second process would load a second copy of the weights and
would not share the in-process lock.

`GET /health` returns `model_loaded: true` only after the MLX weights are
resident. `backend` is `mlx` or `fake`. `weights_loaded` matches
`model_loaded`.

```bash
curl -s http://127.0.0.1:8091/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "weights_loaded": true,
  "backend": "mlx",
  "model": "LiquidAI/d1-2B-MLX-8bit",
  "version": "0.1.0"
}
```

A supervisor such as `launchd` has no terminal. Startup then never downloads
weights. Put the checkpoint on disk first, or set `allow_download`. See
[Configuration](configuration.md).
