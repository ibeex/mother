# mother

Mother is a local terminal chat interface for LLMs.

It supports two modes:

- **Chat mode**: conversational answers only
- **Agent mode**: still conversational, but the model may use tools and then report findings before waiting for your next input

## Features

- terminal-first chat UI
- model switching
- optional thinking display
- session capture with `/save`, `Ctrl+S`, and `mother --save`
- clipboard image paste in the prompt via `Ctrl+V`
- agent mode with tool traces
- guarded local `bash` execution
- web search via Jina Search
- web fetching for public pages, APIs, and localhost services

## Clipboard images

When the prompt input is focused, `Ctrl+V` still behaves like paste, but Mother now
checks the system clipboard for an image first.

- if an image is present, Mother saves it to a temporary file and inserts that path into the prompt
- the pasted image is also attached to the next model request when that path remains in the message
- if no image is present, `Ctrl+V` falls back to normal text paste

Your selected model must support image attachments for multimodal prompts to work.

## Sessions

Each app launch starts a new transient JSONL session under `~/.mother/sessions`.

- `/save` or `Ctrl+S` exports the current session to markdown
- `mother --save` recovers the last unsaved session and exits
- if you quit without saving, the next launch silently deletes that last unsaved JSONL file
- markdown exports include a session summary, prompt context, system prompts, tool calls, tool outputs, and key session events

Markdown exports default to `~/Debian/Documents/mother` when that directory exists,
otherwise `~/Documents/mother`. You can override this with `session_markdown_dir`
in `~/.config/mother/config.toml`.

## Tools in agent mode

Current agent tools:

- `bash`: run local shell commands, guarded by the bash safety classifier
- `web_search`: search the public web using Jina Search
- `web_fetch`: fetch a known URL using either raw HTTP or Jina reader mode

### `web_search`

`web_search` is for when you **do not know the exact URL yet** and want to discover sources.

Implementation notes:

- uses Jina Search API: `https://s.jina.ai/?q=...`
- sends the API key from `pass api/jina`
- returns plain-text search results

If `pass` is not installed or `api/jina` is missing from your password store, search requests will fail with a readable error.

### `web_fetch`

`web_fetch` is for when you **already know the URL** and want to open it.

It supports three modes:

- `auto`: default; chooses the best strategy
- `raw`: direct `urllib` request
- `jina`: fetch through Jina reader for plain-text page extraction

How `auto` behaves:

- uses `raw` for localhost URLs
- uses `raw` for API-style requests with custom method, headers, or body
- uses `jina` for normal public web pages

Jina fetch behavior:

- first tries without auth
- if Jina responds with auth/rate-limit style failure, retries once with the API key from `pass api/jina`

Use `raw` mode when you need:

- localhost access
- exact HTTP behavior
- custom headers
- POST/PUT/etc.
- API calls that should return raw response bodies

## Jina API key setup

Mother expects the Jina API key in your password store at:

```text
api/jina
```

Example:

```bash
pass insert api/jina
```

Then paste your Jina API key as the first line of the secret.

## SSL inspection / custom CA bundle

If your network uses SSL inspection and Python fails TLS verification while `curl` works,
you can configure a custom CA bundle for `web_search` and `web_fetch`.

Config file:

```text
~/.config/mother/config.toml
```

Example:

```toml
tools_enabled = true
ca_bundle_path = "/etc/ssl/certs/ib_cert.pem"
```

Behavior:

- if `ca_bundle_path` is missing or empty, Mother uses only Python/system certificates
- if `ca_bundle_path` is set, that CA bundle is added for `web_search` and `web_fetch`
- if the configured file does not exist, the tool returns a readable error

Mother also relaxes OpenSSL strict X.509 verification for these web tool requests, which helps with some enterprise interception certificates that Python rejects more strictly than `curl`.

## Development

For how to install uv and Python, see [installation.md](installation.md).

For development workflows, see [development.md](development.md).

For instructions on publishing to PyPI, see [publishing.md](publishing.md).
