# mlx-d1

A resident judging server for [LiquidAI d1-2B](https://huggingface.co/LiquidAI/d1-2B).
Send a state, optional images, and questions. Get typed JSON back. The model
stays loaded and does not generate tokens.

The server is not built for one product. A request brings its own state,
images, and questions. Named presets live in the config file. Checking
ComfyUI-generated images is one example preset in `examples/config.toml`, not
built-in logic. The server does not apply a cutoff. Calibrate any threshold
on your own data.

`POST /v1/systemone` follows llama.cpp (`state`, `images`, `questions`). See
the [HTTP API](docs/http-api.md).

## Requirements

- [uv](https://docs.astral.sh/uv/) (or run `mise install` to install the pinned
  version). `uv` manages Python 3.12 automatically from `.python-version`.
- Apple Silicon and `mlx-vlm` 0.7.11 or newer to load the real model. The
  8-bit weights are about 2.22 GB.
- The default model is `LiquidAI/d1-2B-MLX-8bit`. Startup loads a local
  directory or a complete Hugging Face cache entry. It does not download on
  its own. Details are in [Configuration](docs/configuration.md).

The model call sits behind a swappable backend. Tests use a fake backend.
MLX does not run on this repository's Linux CI image.

## Quickstart

```bash
uv sync --extra mlx
cp examples/config.toml mlx-d1.toml
uv run --extra mlx mlx-d1 serve --config mlx-d1.toml
```

The server listens on `127.0.0.1:8091`. There is no authentication.

```bash
curl -s http://127.0.0.1:8091/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"I was charged twice. Please refund one charge.","questions":{"refund":{"type":"noul","instructions":"Is the customer asking for a refund?"}}}'
```

`GET /health` reports whether the MLX weights are resident.

## Documentation

| Topic                                                              | Page                                   |
| ------------------------------------------------------------------ | -------------------------------------- |
| Config, model path priority, cache lookup, and the download prompt | [Configuration](docs/configuration.md) |
| Endpoints and a request and response for every question type       | [HTTP API](docs/http-api.md)           |
| Named question sets, including the sample image preset             | [Presets](docs/presets.md)             |
| File paths, base64, and data URLs                                  | [Images](docs/images.md)               |
| pi and pi-mcp-adapter                                              | [MCP](docs/mcp.md)                     |
| One worker, localhost, and a start with no terminal                | [Running the server](docs/service.md)  |
| Per-request latency, including a llama-server comparison           | [Benchmarking](docs/benchmarking.md)   |
| Tests, the fake backend, and what still needs a Mac                | [Development](docs/development.md)     |

## License

This project is MIT. See [LICENSE](LICENSE).

Model weights are not included. `LiquidAI/d1-2B-MLX-8bit` is under the
LFM Open License v1.0. Read that license on the model card before you use the
weights.
