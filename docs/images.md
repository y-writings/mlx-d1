# Images

An image in a request is a file path, raw base64, or an `image/*;base64` data
URL.

A file path is read only when the resolved file sits inside
`allowed_image_dirs`. Symlinks are resolved before that check. Leave the list
empty to accept only base64 and data URLs.

```toml
allowed_image_dirs = ["/Users/me/Pictures"]
```

`MLX_D1_ALLOWED_IMAGE_DIRS` is split on the platform path separator (`:` on
macOS). Repeat `--allow-dir` to pass the same list on the command line. A
later source replaces the whole list.

A value is treated as base64 when it is at least 64 characters of the base64
alphabet and the bytes look like an image, unless a real file of that name
exists.

`box` and `boxes` questions need exactly one image. That is a model
constraint, not a judging policy. Text-only requests omit `images`. `null`
state with images means the images are the whole state.

Image paths sent through MCP still have to sit under the HTTP server's
allowlist. The MCP process does not open them itself. See [MCP](mcp.md).
