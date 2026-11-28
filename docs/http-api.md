# HTTP API

`POST /v1/systemone` follows llama.cpp
(`/v1/systemone` on the [d1-2B GGUF card](https://huggingface.co/LiquidAI/d1-2B-GGUF)):
`state`, `images`, and `questions`. A client can point the same body at
llama-server. Responses add a `timing` object.

The server listens on `127.0.0.1:8091` by default.

| Method | Path                  | Role                                    |
| ------ | --------------------- | --------------------------------------- |
| GET    | `/health`             | Whether the MLX weights are resident    |
| POST   | `/v1/systemone`       | One state, the questions you send       |
| POST   | `/v1/systemone/batch` | Several states in one model call        |
| POST   | `/v1/compare`         | Two candidates, judged in both orders   |
| GET    | `/v1/presets`         | List named question sets                |
| GET    | `/v1/presets/{name}`  | One preset's questions                  |
| POST   | `/v1/presets/{name}`  | Run a preset. See [Presets](presets.md) |

Image rules are in [Images](images.md). `GET /health` is shown in
[Running the server](service.md).

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
- `box` and `boxes` need exactly one image.
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
