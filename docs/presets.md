# Presets

Presets are named question sets in the config file. They are data. The sample
`image_artifacts` preset asks a few visual questions and tells the model to
ignore ordinary style, lighting, and exposure. Edit those sentences, or delete
the preset. Nothing in the server treats that name specially.

`GET /v1/presets` lists them. `GET /v1/presets/{name}` returns one preset's
questions. `POST /v1/presets/{name}` runs that question set against the state
and images you send. The response is the same envelope as
[`/v1/systemone`](http-api.md).

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

The sample ships in
[`examples/config.toml`](../examples/config.toml).
