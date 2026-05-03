# mother

[![CI](https://github.com/ibeex/mother/actions/workflows/ci.yml/badge.svg)](https://github.com/ibeex/mother/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

Mother is a slow agent for the terminal: a local chat interface for LLMs that works with you one step at a time.

It supports three modes:

- **Chat mode**: conversational answers only
- **Agent mode**: still conversational, but the model may use tools and then report findings before waiting for your next input. Use `/agent` to toggle it on quickly, or `/agent standard` / `/agent conversational` to select it explicitly.
- **Deep research mode**: activated with `/agent deep research` (or via the `/agent` inline profile picker); the model first proposes a research plan, asks for approval or scope changes, then uses only web search/fetch tools in a multi-step loop until the answer is ready

## Why Mother?

Mother's core idea is simple: **slow agent**.

Most coding agents optimize for autonomy. You hand them a task, they plan, chain tools, and come back with an answer or a patch.

Mother deliberately does less.

By default, it is just chat with a few explicit capabilities like `[[fetch ...]]` and guarded shell access. In agent mode, it stays conversational and takes only **one tool step at a time** before handing control back to you.

That slower loop is a feature, not a limitation.

- **Learn while you work**: inspect each step instead of watching a hidden chain unfold
- **Brainstorm interactively**: steer the next move before the agent disappears into execution
- **Stay in control**: change direction, tighten scope, or stop after any tool result
- **See what happened**: actions stay legible instead of collapsing into opaque autonomy
- **Reduce risk**: less unsupervised action on your machine

If tools like Claude Code or Codex are optimized for delegation, Mother is optimized for collaboration.

Mother is not trying to out-autonomize fully agentic coding tools. It is for a different kind of workflow: learning, exploring, debugging, and brainstorming with the model while staying in the loop.

## Features

- terminal-first chat UI
- model switching
- anonymous multi-model `/council` synthesis across configurable members and a configurable judge
- optional thinking display from structured model reasoning when the provider exposes it
- session capture with `/save`, `Ctrl+S`, and `mother --save`
- clipboard image paste in the prompt via `Ctrl+V` (with `Cmd+V` still working for normal paste on macOS)
- explicit `[[fetch https://...]]` prompt expansion in any chat or agent turn
- agent mode with tool traces
- deep research mode with plan-first web research loops
- guarded local `bash` execution
- web search via Jina Search
- web fetching for public pages, APIs, and localhost services

Mother keeps one shared `reasoning_effort` abstraction across providers.

- OpenAI Responses models can also request visible reasoning summaries with
  `openai_reasoning_summary = "detailed"` in `~/.config/mother/config.toml`, and the
  status line shows values like `R medium/detailed`.
- Anthropic reasoning models use the same `reasoning_effort` setting to enable visible
  thinking blocks via provider thinking budgets, and the status line shows values like
  `R medium/thinking`.

## Quick start

### Install

For how to install Python and `uv`, see [installation.md](installation.md).

Install Mother from a local checkout:

```bash
uv tool install --editable .
```

Or install it directly from GitHub:

```bash
uv tool install --from git+https://github.com/ibeex/mother.git mother
```

### Create your config

Mother can scaffold the config file for you:

```bash
mother --init-config
mother --print-config-path
```

This creates `~/.config/mother/config.toml` if it does not already exist.

If you prefer, you can also copy the example files from this repo:

```bash
mkdir -p ~/.config/mother
cp examples/config.toml.example ~/.config/mother/config.toml
cp examples/keys.example.json ~/.config/mother/keys.json
```

### Add at least one model

You can keep secrets out of Git by using either environment variables or
`~/.config/mother/keys.json`.

Example using an environment variable:

```toml
model = "gpt-5"
tools_enabled = true

[[models]]
id = "gpt-5"
name = "gpt-5"
api_type = "openai-responses"
api_key_env = "OPENAI_API_KEY"
supports_tools = true
supports_reasoning = true
supports_images = true
```

`api_type` is important: it tells Mother which provider protocol to use for that model.
If `api_type` is wrong, the model may fail even when `name` and `base_url` are correct.

Valid values are:

- `openai-responses`: for OpenAI Responses-compatible models/endpoints
- `openai-chat`: for OpenAI Chat Completions-compatible models/endpoints, common with local OpenAI-style servers
- `anthropic`: for Anthropic-compatible models/endpoints

This also affects feature wiring such as reasoning/tool behavior, so make sure every
configured model, including `bash_checker`, uses the correct `api_type`.

Example `~/.config/mother/keys.json`:

```json
{
  "OPENAI_KEY": "paste-your-api-key-here"
}
```

Then reference it from `config.toml` with `api_key = "OPENAI_KEY"`.

Example files are included in this repository under:

- `examples/config.toml.example`
- `examples/keys.example.json`

### Bash guard model

If you enable `tools_enabled = true` and want to use the local `bash` tool in agent mode,
add a separate model entry with id `bash_checker`.

Mother uses that model id for the bash safety classifier before any shell command is executed.
The underlying provider/model can be anything you configure, but the id must be `bash_checker`.
If no `bash_checker` model is configured, bash tool calls will be blocked fail-closed.

Example:

```toml
[[models]]
id = "bash_checker"
name = "your-bash-checker-model"
api_type = "openai-chat"
base_url = "http://localhost:1234/v1"
supports_tools = false
supports_reasoning = false
supports_images = false
```

### Run

```bash
mother
```

If you skip `--init-config`, Mother will still create a starter config automatically on first launch.

## Clipboard images

When the prompt input is focused, `Ctrl+V` still behaves like paste, but Mother now
checks the system clipboard for an image first.

On macOS, `Cmd+V` continues to work for regular terminal paste, and `Ctrl+V` now also
works for plain text when no image is present in the clipboard.

- if an image is present, Mother saves it to a temporary file and inserts that path into the prompt
- the pasted image is also attached to the next model request when that path remains in the message
- pasted images are EXIF-orientation corrected, and only resized/optimized when they exceed 2000×2000 or a 4.5MB upload budget
- if no image is present, `Ctrl+V` falls back to normal text paste, including system clipboard text on macOS

Your selected model must support image attachments for multimodal prompts to work.

## Inline web fetch expansion

You can explicitly fetch a URL into the next prompt by writing:

```text
[[fetch https://example.com/page]]
```

Mother fetches the page before sending the turn to the model, injects the fetched
content into the prompt context, and rewrites the user-visible prompt portion to
contain the plain URL.

This works in normal chat turns and in agent-mode turns because it happens before
runtime execution rather than through an LLM tool call.

Notes:

- only explicit `[[fetch ...]]` directives are expanded
- repeated URLs in the same turn are fetched once
- prompt expansion currently limits itself to the first 3 fetched URLs per turn
- fetched content is truncated before injection to avoid exploding prompt size
- fetch failures are included as inline error notes instead of aborting the turn

## Sessions

Each app launch starts a new transient JSONL session under `~/.mother/sessions`.

- `/save` or `Ctrl+S` exports the current session to markdown
- `mother --save` recovers the last unsaved session and exits
- after each markdown export, Mother tries `uv run rumdl fmt --disable MD013 <file>` for cleaner formatting
- if `uv` is not installed, the session is still saved and Mother shows a tip about installing `uv` for better formatting
- if you quit without saving, the next launch silently deletes that last unsaved JSONL file
- markdown exports include a session summary, prompt context, system prompts, tool calls, tool outputs, and key session events

Markdown exports default to `~/Debian/Documents/mother` when that directory exists,
otherwise `~/Documents/mother`. You can override this with `session_markdown_dir`
in `~/.config/mother/config.toml`.

## Council

Use `/council [question]` to run an anonymous multi-model deliberation pass.

What it does:

- sends the council question plus a bounded snapshot of recent conversation context
- includes any pending shell context that would normally be flushed into the next turn
- collects independent answers from the configured council members
- runs anonymous peer review over `Response A`, `Response B`, `Response C`, ...
- asks the configured judge to synthesize the final answer without seeing model ids
- shows live council progress in the reply placeholder while stages 1–3 are running

You can keep the question on the same line, or type just `/council`, press `Enter` to continue on the next line, then press `Ctrl+Enter` to submit the full multiline question.

Important behavior:

- only the stripped council question and the final synthesized answer are added back into the main chat context
- intermediate council drafts, rankings, and reviews stay internal and do not pollute future prompts
- council traces are still inspectable in the TUI and are included in markdown session exports
- `/council` currently ignores image attachments

Example configuration in `~/.config/mother/config.toml`:

```toml
[council]
members = ["gpt-5", "g3", "opus"]
judge = "opus"
max_context_turns = 8
max_context_chars = 12000
```

Notes:

- `members` and `judge` must reference ids from your configured `[[models]]`
- the judge may also be one of the members
- the judge sees anonymous response labels, not model names

## Tools in agent mode

Standard agent tools:

- `bash`: run local shell commands, guarded by the bash safety classifier that uses the configured `bash_checker` model id
- `web_search`: search the public web using Jina Search
- `web_fetch`: fetch a known URL using either raw HTTP or Jina reader mode

Deep research mode keeps only the two web tools:

- `web_search`
- `web_fetch`

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

For release and publishing notes, see [publishing.md](publishing.md).

## Thanks

Inspirations for Mother include:

- [Textual's `examples/mother.py`](https://github.com/Textualize/textual/blob/main/examples/mother.py)
- [`pi-mono`](https://github.com/badlogic/pi-mono)
- [`llm`](https://github.com/simonw/llm)
