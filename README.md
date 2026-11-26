# mlx-d1

A resident judging server for [LiquidAI d1-2B](https://huggingface.co/LiquidAI/d1-2B).
d1 answers typed questions in one forward pass and does not generate tokens.
This process loads the model once, keeps it in memory, and answers HTTP or MCP
requests with JSON.

The server is not built for one product. A request brings its own state,
optional images, and questions. Named presets live in the config file. The
image-artifact preset in `examples/config.toml` is only an example of that
file. The server does not apply a cutoff, a score bar, or a preference of its
own. Calibrate any threshold on your own data.

The HTTP shape of `POST /v1/systemone` follows llama.cpp
(`/v1/systemone` on the [d1-2B GGUF card](https://huggingface.co/LiquidAI/d1-2B-GGUF)):
`state`, `images`, and `questions`. A client can point the same body at
llama-server. Responses add a `timing` object so you can read latency without
a separate probe.

## What runs where

| Piece          | Role                                                          |
| -------------- | ------------------------------------------------------------- |
| `mlx-d1 serve` | Loads `LiquidAI/d1-2B-MLX-8bit` once and serves HTTP          |
| `mlx-d1 mcp`   | Stdio MCP tools that call the HTTP server                     |
| `mlx-d1 bench` | Measures per-request latency, optionally against llama-server |

Target machine: a Mac Studio (Apple Silicon), often beside LM Studio. The
default bind is `127.0.0.1:8091` so it does not take llama-server's or LM
Studio's port. There is no authentication. Do not expose the port on a shared
network.

MLX cannot run on a non-Apple machine. The model call sits behind a backend
interface (`mlx_d1.backends`). Tests use a fake backend. See
[What was verified](#what-was-verified) before trusting answers from weights.

## Setup on macOS Apple Silicon

Use Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[mlx]"
```

`mlx-vlm` must be at least 0.7.11. The 8-bit weights are about 2.22 GB.
Startup loads them only from a local directory or from a complete copy already
in the Hugging Face cache. It does not start a download on its own.

```bash
cp examples/config.toml mlx-d1.toml
```

Edit `allowed_image_dirs` if clients will send file paths. Leave it empty to
accept only base64 and data URLs. Then:

```bash
mlx-d1 serve --config mlx-d1.toml
```

Useful overrides, highest priority last: the TOML file, then environment
variables, then flags.

| Setting                   | Flag                       | Environment variable        |
| ------------------------- | -------------------------- | --------------------------- |
| Model id                  | `--model`                  | `MLX_D1_MODEL`              |
| Bind address              | `--host`                   | `MLX_D1_HOST`               |
| Port                      | `--port`                   | `MLX_D1_PORT`               |
| Backend `mlx` or `fake`   | `--backend`                | `MLX_D1_BACKEND`            |
| Score readout             | `--score-readout`          | `MLX_D1_SCORE_READOUT`      |
| Allowed image directories | `--allow-dir` (repeatable) | `MLX_D1_ALLOWED_IMAGE_DIRS` |
| Config file path          | `--config`                 | `MLX_D1_CONFIG`             |
| Download without asking   | `--allow-download`         | `MLX_D1_ALLOW_DOWNLOAD`     |

`MLX_D1_ALLOWED_IMAGE_DIRS` is split on the platform path separator (`:` on
macOS). `MLX_D1_SET_SIZE` sets the d1 `set_size` (empty, `none`, or `null`
keeps the model default). A `mlx-d1.toml` in the working directory is used
when you do not pass `--config`.

`GET /health` returns `model_loaded: true` only after the MLX weights are
resident. `backend` is `mlx` or `fake`.

## Where the weights come from

`mlx-d1 serve` is the command that loads the model. `mlx-d1 bench` and
`mlx-d1 mcp` do not. They talk to a server that is already running.

On startup the loader looks for the configured model:

- A local directory that contains `config.json` and a `.safetensors` file is
  loaded as-is. There is no prompt and no network download.
- A Hugging Face repo id that is already complete in the local cache (the same
  files `mlx-vlm` would use) is loaded the same way. The cache lookup asks the
  Hub library for local files only.
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

The same priority applies as for the model itself: the TOML file, then the
environment variable, then the flag. A later source replaces an earlier one.

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

The loader then calls `mlx_vlm.load` with that directory only, and sets
`HF_HUB_OFFLINE=1` for the call. A repo id is never passed to `load`, so a
missing file cannot fall through into a download.

Check it:

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

Run one worker. A second process would load a second copy of the weights and
would not share the in-process lock.

## Request rules

- `state` is a string, any JSON value, or `null`. `null` with images means the
  images are the whole state. Text-only requests omit `images`.
- `questions` is a non-empty object. Names are the keys. Every question needs
  `type` and `instructions`. Unknown question fields are forwarded to the
  model. `bool` is accepted as an alias of `noul`.
- Supported types are the open d1-2B card: `noul`, `choice` (including
  `multiple: true`), `score`, `text`, `number`, `date`, `number` with `range`
  and `step`, `box`, and `boxes`. `spans` is a Liquid API type and is not
  accepted here.
- `choice` `criteria` is an object of option name to description (`null`
  becomes an empty description). At least two options.
- `score` `criteria` is 2 to 10 strings, lowest first.
- `number` `range` is `[low, high]` with `low < high`. `step`, when set, must
  be positive and must divide the span. The model card still coarsens a range
  to at most 128 values.
- `boxes` `max` is an integer from 1 to 16.
- `box` and `boxes` need exactly one image. That is a model constraint, not a
  judging policy.
- An image is a file path, raw base64, or an `image/*;base64` data URL. A file
  path is read only when the resolved file sits inside `allowed_image_dirs`.
  Symlinks are resolved before that check. A value is treated as base64 when
  it is at least 64 characters of the base64 alphabet and the bytes look like
  an image, unless a real file of that name exists.
- `score_readout` is `median` (default), `mean`, or `argmax`. `set_size` is a
  positive integer or omitted for the model default.
- Inference is serial. Image decoding happens before the model lock.
  `system_one` serves one request. `system_one_batch` serves a batch and the
  two orders of a comparison.

`timing.latency_ms` is server time for that HTTP call, including image loads.
`timing.inference_ms` is the model call only. `usage.output_tokens` is 0.

The answer objects below are the shapes from the d1 model card. They are not
measurements from this repository. A value is missing when `found` is under
0.5, and the value is then `null`.

## noul

Yes or no, as `noul`: the probability of yes.

```json
{
  "state": "I was charged twice and my parcel never arrived. Please refund one charge.",
  "questions": {
    "refund": {
      "type": "noul",
      "instructions": "Is the customer asking for a refund?"
    }
  }
}
```

```bash
curl -s http://127.0.0.1:8091/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @- <<'JSON'
{
  "state": "I was charged twice and my parcel never arrived. Please refund one charge.",
  "questions": {
    "refund": {
      "type": "noul",
      "instructions": "Is the customer asking for a refund?"
    }
  }
}
JSON
```

```json
{
  "model": "LiquidAI/d1-2B-MLX-8bit",
  "answers": {
    "refund": { "type": "noul", "noul": 1.0 }
  },
  "usage": { "input_tokens": 189, "output_tokens": 0 },
  "timing": { "latency_ms": 18.6, "inference_ms": 18.2 }
}
```

## choice

One option. `criteria` maps each option name to a description. `probabilities`
sums to 1. `confidence` is how clearly one option leads.

```json
{
  "state": "I was charged twice and my parcel never arrived. Please refund one charge.",
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Charges, refunds, invoices",
        "shipping": "Delivery and parcels",
        "technical": "App or site faults"
      }
    }
  }
}
```

```json
{
  "model": "LiquidAI/d1-2B-MLX-8bit",
  "answers": {
    "team": {
      "type": "choice",
      "choice": "billing",
      "probabilities": { "billing": 0.97, "shipping": 0.02, "technical": 0.01 },
      "confidence": 0.96
    }
  },
  "usage": { "input_tokens": 189, "output_tokens": 0 },
  "timing": { "latency_ms": 20.0, "inference_ms": 19.4 }
}
```

## choice with multiple

Each option is its own probability of yes. They do not sum to 1. `choices`
lists options at or above 0.5.

```json
{
  "state": "I was charged twice and my parcel never arrived. Please refund one charge.",
  "questions": {
    "topics": {
      "type": "choice",
      "multiple": true,
      "instructions": "Which topics does the message raise?",
      "criteria": {
        "billing": "Payments and charges",
        "delivery": "Shipping and parcels",
        "login": "Signing in to the account"
      }
    }
  }
}
```

```json
{
  "model": "LiquidAI/d1-2B-MLX-8bit",
  "answers": {
    "topics": {
      "type": "choice",
      "choices": ["billing", "delivery"],
      "probabilities": {
        "billing": 0.968,
        "delivery": 0.969,
        "login": 0.001
      }
    }
  },
  "usage": { "input_tokens": 189, "output_tokens": 0 },
  "timing": { "latency_ms": 21.0, "inference_ms": 20.4 }
}
```

## score

`criteria` lists 2 to 10 levels, lowest first. `score` is the readout on that
scale (`median` by default, so it lands on a level index). `legend` maps the
index back to your text.

```json
{
  "state": "I was charged twice and my parcel never arrived. Please refund one charge.",
  "questions": {
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this ticket?",
      "criteria": ["low", "medium", "high"]
    }
  }
}
```

```json
{
  "model": "LiquidAI/d1-2B-MLX-8bit",
  "answers": {
    "urgency": {
      "type": "score",
      "score": 1,
      "confidence": 0.4,
      "probabilities": { "0": 0.1, "1": 0.7, "2": 0.2 },
      "legend": { "0": "low", "1": "medium", "2": "high" }
    }
  },
  "usage": { "input_tokens": 189, "output_tokens": 0 },
  "timing": { "latency_ms": 19.0, "inference_ms": 18.5 }
}
```

## text

Reads a string from the state. `span` is `[start, end)` in the rendered state
(the string itself, or pretty-printed JSON when `state` is not a string).
`found` is the probability the value is present.

```json
{
  "state": "ACME SA\nInvoice 2026-117\nTotal due: 1 234,50 EUR\nPay before 03/04/2026 (day/month/year)",
  "questions": {
    "payee": { "type": "text", "instructions": "Who issued the invoice?" }
  }
}
```

```json
{
  "model": "LiquidAI/d1-2B-MLX-8bit",
  "answers": {
    "payee": {
      "type": "text",
      "text": "ACME SA",
      "span": [0, 7],
      "found": 0.986
    }
  },
  "usage": { "input_tokens": 80, "output_tokens": 0 },
  "timing": { "latency_ms": 17.0, "inference_ms": 16.4 }
}
```

## number

Reads one number from the state, plus the written form and an optional unit.
This is extraction, not an estimate. A missing number has `number: null` and a
low `found`.

```json
{
  "state": "ACME SA\nInvoice 2026-117\nTotal due: 1 234,50 EUR\nPay before 03/04/2026 (day/month/year)",
  "questions": {
    "total": { "type": "number", "instructions": "How much is due?" }
  }
}
```

```json
{
  "model": "LiquidAI/d1-2B-MLX-8bit",
  "answers": {
    "total": {
      "type": "number",
      "number": 1234.5,
      "unit": "EUR",
      "text": "1 234,50 EUR",
      "span": [36, 48],
      "found": 1
    }
  },
  "usage": { "input_tokens": 80, "output_tokens": 0 },
  "timing": { "latency_ms": 17.2, "inference_ms": 16.6 }
}
```

## date

The date is ISO 8601. `text` is the span as written.

```json
{
  "state": "ACME SA\nInvoice 2026-117\nTotal due: 1 234,50 EUR\nPay before 03/04/2026 (day/month/year)",
  "questions": {
    "due": { "type": "date", "instructions": "By when must it be paid?" }
  }
}
```

```json
{
  "model": "LiquidAI/d1-2B-MLX-8bit",
  "answers": {
    "due": {
      "type": "date",
      "date": "2026-04-03",
      "text": "03/04/2026",
      "span": [60, 70],
      "found": 1
    }
  },
  "usage": { "input_tokens": 80, "output_tokens": 0 },
  "timing": { "latency_ms": 17.1, "inference_ms": 16.5 }
}
```

## number with range and step

An estimate. `number` is the median. `interval` is `[p10, p90]`. There is no
`found`. Use this for counts and scores. The picture below is a 1 pixel PNG so
the example is self-contained; point `images` at your own file or data URL.

```json
{
  "state": null,
  "images": [
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
  ],
  "questions": {
    "cats": {
      "type": "number",
      "instructions": "How many cats are there?",
      "range": [0, 10],
      "step": 1
    }
  }
}
```

```json
{
  "model": "LiquidAI/d1-2B-MLX-8bit",
  "answers": {
    "cats": { "type": "number", "number": 2, "interval": [2, 2], "unit": null }
  },
  "usage": { "input_tokens": 40, "output_tokens": 0 },
  "timing": { "latency_ms": 45.0, "inference_ms": 44.0 }
}
```

## box

One object in the single picture. Edges are fractions of the picture:
`top`, `bottom`, `left`, `right`. `box` is `null` when `found` is under 0.5.

```json
{
  "state": null,
  "images": [
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
  ],
  "questions": {
    "remote": {
      "type": "box",
      "instructions": "Where is the remote control on the left?"
    }
  }
}
```

```json
{
  "model": "LiquidAI/d1-2B-MLX-8bit",
  "answers": {
    "remote": {
      "type": "box",
      "box": { "top": 0.15, "bottom": 0.24, "left": 0.06, "right": 0.28 },
      "found": 0.9,
      "confidence": 0.7
    }
  },
  "usage": { "input_tokens": 40, "output_tokens": 0 },
  "timing": { "latency_ms": 53.0, "inference_ms": 52.0 }
}
```

## boxes

Every instance, at most `max` (1 to 16, default 16). `count` is how many were
at or above 0.5. `at_least` is true when every slot was used, so the picture
may hold more. `presence` is P(count >= k) for each slot. Boxes are ordered by
left edge, then top.

```json
{
  "state": null,
  "images": [
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
  ],
  "questions": {
    "every_cat": { "type": "boxes", "instructions": "Every cat.", "max": 4 }
  }
}
```

```json
{
  "model": "LiquidAI/d1-2B-MLX-8bit",
  "answers": {
    "every_cat": {
      "type": "boxes",
      "boxes": [
        {
          "box": { "top": 0.1, "bottom": 0.5, "left": 0.05, "right": 0.4 },
          "found": 0.99,
          "confidence": 0.8
        }
      ],
      "count": 2,
      "at_least": false,
      "presence": [0.99, 0.97, 0.2, 0.05]
    }
  },
  "usage": { "input_tokens": 40, "output_tokens": 0 },
  "timing": { "latency_ms": 100.0, "inference_ms": 98.0 }
}
```

## Batch

Several states in one model call:

```json
{
  "requests": [
    {
      "state": "Where is my parcel?",
      "questions": {
        "team": {
          "type": "choice",
          "instructions": "Which team?",
          "criteria": { "billing": "Charges", "shipping": "Delivery" }
        }
      }
    },
    {
      "state": "The app crashes when I open settings.",
      "questions": {
        "team": {
          "type": "choice",
          "instructions": "Which team?",
          "criteria": { "billing": "Charges", "shipping": "Delivery" }
        }
      }
    }
  ]
}
```

`POST /v1/systemone/batch` returns `results` in the same order, one
`answers` / `usage` object each, plus a summed `usage` and one `timing` for
the whole call.

## Compare

`POST /v1/compare` judges two candidates on criteria you send. Each candidate
has `text`, `images`, or both. `criteria` is one instruction, a list of
instructions, or a list of `{name, instructions}` objects. Optional `state` is
shared context.

To counter position bias the server asks twice. The first call labels A as
FIRST and B as SECOND. The second call swaps them. A side that wins both
orders wins that criterion. A split, or an answer that is not FIRST or
SECOND, is a tie. With several criteria, the overall `winner` is a strict
majority of criterion winners (`a`, `b`, or `tie`). Ties do not vote.
Probabilities are returned for each order and are not averaged. No cutoff is
applied: a low-confidence double win is still a win, and you decide what to
do with `confidence`.

```json
{
  "state": "Keep the picture that better matches the prompt.",
  "a": {
    "text": "Prompt: a red cup on a table.",
    "images": [
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    ]
  },
  "b": {
    "text": "Prompt: a red cup on a table.",
    "images": [
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    ]
  },
  "criteria": [
    {
      "name": "prompt_match",
      "instructions": "Which picture matches the prompt more closely?"
    }
  ]
}
```

```json
{
  "model": "LiquidAI/d1-2B-MLX-8bit",
  "winner": "a",
  "counts": { "a": 1, "b": 0, "tie": 0 },
  "criteria": {
    "prompt_match": {
      "winner": "a",
      "forward": {
        "order": ["a", "b"],
        "choice": "FIRST",
        "mapped": "a",
        "probabilities": { "FIRST": 0.8, "SECOND": 0.2 },
        "confidence": 0.6,
        "answer": { "type": "choice", "choice": "FIRST" }
      },
      "reversed": {
        "order": ["b", "a"],
        "choice": "SECOND",
        "mapped": "a",
        "probabilities": { "FIRST": 0.3, "SECOND": 0.7 },
        "confidence": 0.4,
        "answer": { "type": "choice", "choice": "SECOND" }
      }
    }
  },
  "usage": { "input_tokens": 400, "output_tokens": 0 },
  "timing": { "latency_ms": 80.0, "inference_ms": 78.0 }
}
```

Text-only compare works the same way: omit `images` and put the two texts in
`a.text` and `b.text`.

## Presets

Presets are named question sets in the config file. They are data. The sample
`image_artifacts` preset asks a few visual questions and tells the model to
ignore ordinary style, lighting, and exposure. Edit those sentences, or delete
the preset. Nothing in the server treats that name specially.

```bash
curl -s http://127.0.0.1:8091/v1/presets
curl -s http://127.0.0.1:8091/v1/presets/image_artifacts \
  -H 'Content-Type: application/json' \
  -d '{"state": null, "images": ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"]}'
```

```json
{
  "state": null,
  "images": [
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
  ]
}
```

The response is the same envelope as `/v1/systemone`.

## MCP from pi

Start the HTTP server first. In another terminal, the MCP process only
forwards tool calls:

```bash
mlx-d1 mcp
```

`MLX_D1_URL` defaults to `http://127.0.0.1:8091`.

Tools:

| Tool         | Call                                         |
| ------------ | -------------------------------------------- |
| `decide`     | `decide(questions, state=None, images=None)` |
| `compare`    | `compare(a, b, criteria, state=None)`        |
| `run_preset` | `run_preset(name, images=None, state=None)`  |

`a` and `b` are objects with optional `text` and `images`. `criteria` is a
list of strings or `{name, instructions}` objects. `state` may be a string, an
object, or a list. For a JSON state from MCP, pass the object rather than a
string of JSON so d1 can attach a JSON pointer to extracted values.

pi-mcp-adapter reads `.mcp.json` (project) or `~/.config/mcp/mcp.json`. Pi
0.99 and later also reads `.pi/mcp.json`.

```json
{
  "mcpServers": {
    "mlx-d1": {
      "command": "mlx-d1",
      "args": ["mcp"],
      "env": { "MLX_D1_URL": "http://127.0.0.1:8091" }
    }
  }
}
```

Or, from a project directory, with the virtualenv's executable on `PATH`:

```bash
pi mcp add -l mlx-d1 -- mlx-d1 mcp
```

Image paths in tool arguments still have to sit under the HTTP server's
allowlist. The MCP process does not open them itself.

## Latency

Every response includes `timing`. For a repeatable record, point `bench` at a
running server. Warmup calls are excluded.

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

## Development

On any platform, including this Linux CI image:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

`mlx-d1 serve --backend fake` answers without weights so you can exercise the
HTTP surface. Fake answers are placeholders.

To add a backend, implement `system_one` and `system_one_batch` on the
protocol in `mlx_d1.backends.base` and construct it in `mlx_d1.factory`. Media
items are PIL images. Do not call the implementation concurrently. The MLX
class also holds a thread lock, because two Metal forwards at once are not
supported.

## What was verified

Automated tests, run here and in CI, use the fake backend and a stand-in for
`mlx_vlm.load`. They cover:

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
  `load` is called with a local directory while `HF_HUB_OFFLINE=1`

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

## License

This project is MIT. See [LICENSE](LICENSE).

Model weights are not included. `LiquidAI/d1-2B-MLX-8bit` is under the
LFM Open License v1.0. Read that license on the model card before you use the
weights.
