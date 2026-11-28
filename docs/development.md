# Development

On any platform, including this repository's Linux CI image:

```bash
uv sync
uv run pytest
```

`uv run mlx-d1 serve --backend fake` answers without weights so you can
exercise the HTTP surface. Fake answers are placeholders.

To add a backend, implement `system_one` and `system_one_batch` on the
protocol in `mlx_d1.backends.base` and construct it in `mlx_d1.factory`. Media
items are PIL images. Do not call the implementation concurrently. The MLX
class also holds a thread lock, because two Metal forwards at once are not
supported.

## What was verified

Automated tests use the fake backend and a stand-in for `mlx_vlm.load`. They
cover:

- the `/v1/systemone` envelope and a request for every supported question type
- question passthrough, including the `bool` alias
- compare order swap, a double win, a split tie, and majority across criteria
- path allowlist, sibling-prefix rejection, symlink escape, data URLs, and
  raw base64
- one in-flight model call across systemone, compare, and presets
- presets loaded from `examples/config.toml`
- a live process that keeps answering JSON
- the MLX adapter's call into `system_one` / `system_one_batch` and its lock
- local-only weight resolution: a directory or a cached repo does not
  download, a declined or non-interactive start does not download, and
  `load` is called with a local directory while the Hub client is offline

They do not load `LiquidAI/d1-2B-MLX-8bit` and they do not check answer
quality.

On the Mac, with the real weights, check:

- `mlx-d1 serve` stays up and `model_loaded` is true
- a model already on disk or in the Hugging Face cache starts with no prompt
  and no download; a missing repo id asks first; a start with no terminal
  does not download unless `allow_download` is set
- a curl of each question type, plus compare, a preset, and a second request
  from another process
- `mlx-d1 bench`, and the llama-server comparison if you want that record
- `mlx-d1 mcp` from pi
- your own review of real A/B pairs and any cutoff you want to apply

Reviewing 20 to 50 generated pairs for bias, including whether brighter images
are scored unfairly, is out of scope for this repository. The sample preset
only asks the model, in its question text, to ignore ordinary exposure.
