# Configuration

[mlx-d1](../README.md) reads settings from defaults, then a TOML file, then
environment variables, then flags. A later source replaces an earlier one.

`mlx-d1 serve` loads the model. `mlx-d1 bench` and `mlx-d1 mcp` do not. They
talk to a server that is already running.

```bash
cp examples/config.toml mlx-d1.toml
mlx-d1 serve --config mlx-d1.toml
```

A `mlx-d1.toml` in the working directory is used when you do not pass
`--config` and `MLX_D1_CONFIG` is unset.

## Settings

| Setting                   | Flag                       | Environment variable        |
| ------------------------- | -------------------------- | --------------------------- |
| Model id or directory     | `--model`                  | `MLX_D1_MODEL`              |
| Bind address              | `--host`                   | `MLX_D1_HOST`               |
| Port                      | `--port`                   | `MLX_D1_PORT`               |
| Backend `mlx` or `fake`   | `--backend`                | `MLX_D1_BACKEND`            |
| Score readout             | `--score-readout`          | `MLX_D1_SCORE_READOUT`      |
| Allowed image directories | `--allow-dir` (repeatable) | `MLX_D1_ALLOWED_IMAGE_DIRS` |
| Config file path          | `--config`                 | `MLX_D1_CONFIG`             |
| Download without asking   | `--allow-download`         | `MLX_D1_ALLOW_DOWNLOAD`     |

`MLX_D1_ALLOWED_IMAGE_DIRS` is split on the platform path separator (`:` on
macOS). See [Images](images.md).

`MLX_D1_SET_SIZE` sets the d1 `set_size`. An empty value, `none`, or `null`
keeps the model default. There is no flag for it.

`score_readout` is `median` (default), `mean`, or `argmax`.

## Where the weights come from

On startup the loader looks for the configured model:

- A local directory that contains `config.json` and a `.safetensors` file is
  loaded as-is. There is no prompt and no network download.
- A Hugging Face repo id that is already a complete checkpoint in the local
  cache (the same files `mlx-vlm` would use) is loaded the same way. The cache
  lookup asks the Hub library for local files only. A partial cache, such as a
  download that stopped early, is treated as missing so you can confirm a
  download or pass `--allow-download` to finish it.
- Otherwise the program prints the repo id and a size when it can (the model
  card's figure for `LiquidAI/d1-2B-MLX-8bit` is about 2.22 GB) and asks
  `Download it now? [y/N]`. `y` or `yes` downloads, then loads. Any other
  answer, including Enter, does not download.

A relative folder has to exist, or be written as `./folder`, so a missing path
is not treated as a repo id.

The answer has to come from a terminal. `launchd`, an MCP client, or any other
start with no TTY never downloads. Opt in for those cases with one of:

- `allow_download = true` in `mlx-d1.toml`
- `MLX_D1_ALLOW_DOWNLOAD=1`
- `mlx-d1 serve --allow-download`

That opt-in uses the same priority as the model path: the TOML file, then the
environment variable, then the flag.

If you decline, or there is no terminal, the process exits without fetching
weights and prints a manual download, for example:

```bash
huggingface-cli download LiquidAI/d1-2B-MLX-8bit --local-dir ./d1-2B-MLX-8bit
```

Point mlx-d1 at that folder with one of these. Later sources replace earlier
ones:

1. `model` in `mlx-d1.toml`
2. `MLX_D1_MODEL`
3. `--model`

```bash
# mlx-d1.toml
# model = "/absolute/path/to/d1-2B-MLX-8bit"

MLX_D1_MODEL=/absolute/path/to/d1-2B-MLX-8bit mlx-d1 serve
mlx-d1 serve --model /absolute/path/to/d1-2B-MLX-8bit
```

The loader then calls `mlx_vlm.load` with that directory only. For the call it
also turns on the Hub client's offline flag, including the copy
`huggingface_hub` captured when it was imported, so a later file lookup cannot
reach the network. A repo id is never passed to `load`.
