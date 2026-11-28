# MCP

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
allowlist. The MCP process does not open them itself. See [Images](images.md).

The MCP process does not load the model. A client that starts `mlx-d1 serve`
with no terminal does not download weights. See
[Configuration](configuration.md).
