# Bash Tool Design Notes

This document distills how Pi handles shell execution and turns it into a generic design that can be implemented in another project, including a Python-based one.

It is intentionally library-agnostic. The goal is to preserve behavior, not TypeScript APIs.

## 1. What Pi actually does

Pi has two related shell-execution paths:

1. **LLM-invoked bash tool**
   - The model calls a tool named `bash`.
   - Input:
     - `command: string`
     - `timeout?: number` in seconds
   - Output:
     - combined stdout/stderr as text
     - tail-truncated when too large
     - optional metadata about truncation and temp-file location
   - Non-zero exit code is treated as a tool failure.

2. **Direct user bash commands in interactive mode**
   - User types `!command` or `!!command`.
   - `!command` executes immediately and is usually added to conversation history.
   - `!!command` executes immediately but is excluded from LLM context.
   - Pi stores these as structured `bashExecution` messages.

The two paths share the same overall model:
- execute command in a shell
- stream output while running
- support cancellation
- truncate large output from the tail
- preserve full output in a temp file when needed
- optionally record the result into conversation/session history

## 2. External contract for a generic bash tool

A good generic contract is:

### Input

- `command: string` — required shell command
- `timeout: number | null` — optional timeout in seconds

### Success result

- `content` — human-readable text returned to the agent/model
- `details` — optional structured metadata, for UI/logging only
  - `truncation`
  - `full_output_path`

### Failure result

Failures should still include useful captured output when possible.

Typical failure cases:
- shell could not be started
- working directory does not exist
- command timed out
- command was aborted/cancelled
- command exited with non-zero status

Pi treats non-zero exit as a failure, not as a successful result with an exit code.

## 3. Execution model

### 3.1 Run through a shell

Pi executes commands through a shell rather than trying to parse commands itself.

Generic behavior:
- resolve a shell executable
- invoke shell with `-c <command>` or equivalent
- set working directory
- inherit environment, then apply tool-specific overrides
- pipe both stdout and stderr
- track the process so the entire process tree can be killed

This is important because users and models expect shell syntax to work:
- pipes
- redirects
- `&&`, `||`
- env vars
- shell builtins
- multiline commands

### 3.2 Working directory

The tool is created/configured for a specific working directory.

If the working directory does not exist, fail before spawning.

### 3.3 Environment

Pi starts from the current process environment and may prepend additional tool directories to `PATH`.

A generic implementation should support:
- base inherited environment
- optional environment patching
- optional command prefixing
- optional spawn hook that can modify:
  - command
  - cwd
  - env

This makes the tool adaptable to:
- SSH
- containers
- chroots
- sandboxes
- CI wrappers

## 4. Streaming behavior

Pi streams output while the command runs.

Generic behavior:
1. read stdout and stderr incrementally
2. append chunks to an in-memory rolling buffer
3. optionally write chunks to a temp file after output crosses a size threshold
4. emit partial updates to the UI/agent during execution

The important design point is this:

**Partial updates should be based on a rolling tail, not unbounded accumulated output.**

Otherwise a long-running command can flood the UI or the agent context.

## 5. Output policy: combine stdout and stderr

Pi merges stdout and stderr into one text stream for the returned result.

That is a good default for agent tools because:
- models usually need the full observable outcome
- many CLI tools write useful diagnostics to stderr
- agent reasoning is simpler with one combined transcript

If your project needs exact stream separation, keep both internally, but still consider returning a merged human-readable view.

## 6. Truncation policy

This is one of the most important parts.

Pi requires tool output truncation to protect context size.

### 6.1 Limits used by Pi

- **2000 lines**
- **50 KB**
- whichever limit is hit first

### 6.2 Why tail truncation

Pi uses **tail truncation** for bash output.

That is the right choice for shell commands because the end of output usually contains the most useful information:
- final status
- error messages
- last log lines
- summary lines

For file reads/search results, head truncation often makes more sense. For bash output, tail truncation is the better default.

### 6.3 Full output preservation

When output exceeds the byte threshold, Pi starts writing all output to a temp file.

Important detail:
- it does **not** wait until the command finishes
- once threshold is crossed, it opens a temp file and continues streaming all subsequent output there
- it also writes already-buffered output into the temp file so the file contains the full transcript

### 6.4 Rolling buffer size

Pi keeps a rolling in-memory buffer larger than the final output limit.

Why:
- final result needs the last N lines / last N bytes
- you do not want to reread the full temp file just to compute the tail
- a buffer somewhat larger than the display limit gives enough data to compute the final truncated tail cleanly

A practical rule:
- final limit: 50 KB
- rolling buffer: about 2x that

### 6.5 Special truncation edge case

Pi handles the case where a single last line is itself larger than the byte limit.

In that case:
- keep only the last bytes of that line
- ensure truncation respects UTF-8 boundaries
- mark that the kept line is partial

This prevents returning an empty result for very large single-line outputs.

### 6.6 User-facing truncation notice

Pi appends a clear notice when output is truncated, including the path to the temp file.

Generic examples:
- `[Showing lines X-Y of N. Full output: /tmp/... ]`
- `[Showing lines X-Y of N (50KB limit). Full output: /tmp/... ]`
- `[Showing last 50KB of line N. Full output: /tmp/... ]`

This is good behavior because the model can decide to inspect the full output later if needed.

## 7. Timeout and cancellation

### 7.1 Timeout

Pi supports optional timeout in seconds.

Behavior:
- start a timer when execution begins
- on timeout, kill the entire process tree
- return failure with captured partial output plus a timeout message

Generic failure text:
- `Command timed out after <n> seconds`

### 7.2 Cancellation

Pi supports abort/cancel through an external signal.

Behavior:
- if aborted while running, kill the process tree
- preserve any partial output already captured
- return failure with an abort message

Generic failure text:
- `Command aborted`

### 7.3 Kill the whole process tree

This is critical.

Do not kill only the direct shell process if child processes may survive.

Your implementation should kill:
- shell
- child processes spawned by the shell
- grandchildren if possible

The exact mechanism depends on platform and runtime, but the requirement is generic.

## 8. Exit-code handling

Pi treats command exit status like this:

- `0` → success
- non-zero → failure, include output and append `Command exited with code <n>`
- timeout/abort → failure with special message

This is the right default for an LLM tool because non-zero exit usually means the model should reconsider and recover.

## 9. Normalization and sanitization

Pi differs slightly by path:

- **LLM bash tool path**: captures bytes and decodes to UTF-8 text for tool results
- **interactive direct-bash path**: additionally sanitizes for terminal display
  - strips ANSI escape codes
  - normalizes line endings
  - removes problematic binary/control characters

For a new implementation, the safest generic choice is:
- keep raw bytes internally if needed
- decode to text carefully
- sanitize anything shown to humans or sent into model context

Recommended sanitization:
- normalize `\r\n` and `\r` to `\n`
- strip ANSI if the consumer is not a terminal emulator
- remove binary/control garbage except tab/newline
- avoid breaking UTF-8 sequences when truncating

## 10. Structured result model

Pi uses a structured result object for direct command execution.

A generic form is:

```text
BashResult
- output: string
- exit_code: int | None
- cancelled: bool
- truncated: bool
- full_output_path: string | None
```

This is a useful internal format even if your tool API returns a different envelope.

## 11. Extensibility hooks

Pi makes bash execution pluggable in two ways.

### 11.1 Execution backend abstraction

Instead of hardwiring local process spawning, define an execution interface such as:

```text
exec(command, cwd, on_data, signal, timeout, env) -> exit_code
```

That allows swapping in:
- local shell execution
- SSH execution
- container execution
- sandboxed execution
- remote worker execution

### 11.2 Spawn hook / context rewrite

Pi also supports a hook that can rewrite the spawn context before execution:
- command
- cwd
- env

This is very useful for:
- sourcing profiles
- changing root directories
- adding env vars
- enabling aliases
- wrapping commands in a sandbox runner

## 12. Conversation integration for an agent product

If your project is an agent, not just a shell wrapper, Pi has an additional design worth copying.

### 12.1 Persist bash executions as structured messages

Pi stores direct user shell commands as structured messages like:

```text
bashExecution
- command
- output
- exit_code
- cancelled
- truncated
- full_output_path
- timestamp
- exclude_from_context
```

### 12.2 Convert them into LLM context only when needed

When Pi includes these messages in model context, it converts them into a user-style text block roughly like:

```text
Ran `git status`
```
<output here>
```

Command exited with code 1
[Output truncated. Full output: /tmp/...]
```

This is a good pattern because:
- session history stays structured
- LLM context stays simple text
- the tool/runtime can choose to omit some shell events from context

### 12.3 Support commands excluded from context

Pi supports a direct-bash mode equivalent to:
- `!command` → include in future LLM context
- `!!command` → do not include in future LLM context

This is a useful feature for:
- exploratory shell commands
- commands with noisy output
- commands containing sensitive content

## 13. Ordering rules while the agent is already streaming

This is a subtle but important Pi behavior.

If a direct user shell command runs while the model is still in the middle of a turn, Pi does **not** immediately insert that shell result into conversation history.

Reason:
- the model may already have emitted tool calls
- many providers require strict `tool_call -> tool_result` ordering
- inserting an unrelated user message in the middle can break protocol validity

So Pi does this instead:
1. execute the shell command immediately
2. show the output in a pending UI area
3. store the shell result in a pending list
4. flush it into chat/history just before the next normal user prompt

If your project has streamed tool-calling conversations, copy this rule.

## 14. Suggested generic architecture

A clean implementation separates three layers.

### Layer 1: executor

Responsible for:
- starting the shell process
- streaming stdout/stderr
- timeout/cancel handling
- process tree termination

### Layer 2: capture/truncation manager

Responsible for:
- rolling output buffer
- temp file creation when threshold is crossed
- text decoding/sanitization
- final tail truncation
- metadata about truncation

### Layer 3: agent/tool wrapper

Responsible for:
- validating tool input
- returning success/failure in the tool protocol
- emitting streaming updates to UI or agent runtime
- converting shell executions into conversation/session messages
- applying policy such as exclude-from-context

## 15. Python-oriented pseudocode

This is intentionally generic pseudocode, not tied to a specific async/process library.

```python
class BashResult:
    def __init__(self, output, exit_code, cancelled, truncated, full_output_path=None):
        self.output = output
        self.exit_code = exit_code
        self.cancelled = cancelled
        self.truncated = truncated
        self.full_output_path = full_output_path


class OutputCapture:
    def __init__(self, max_lines=2000, max_bytes=50 * 1024, rolling_bytes=100 * 1024):
        self.max_lines = max_lines
        self.max_bytes = max_bytes
        self.rolling_bytes = rolling_bytes
        self.total_bytes = 0
        self.chunks = []
        self.chunks_bytes = 0
        self.temp_file = None

    def add_bytes(self, raw_bytes):
        self.total_bytes += len(raw_bytes)
        text = decode_and_sanitize(raw_bytes)

        if self.total_bytes > self.max_bytes and self.temp_file is None:
            self.temp_file = open_temp_output_file()
            for chunk in self.chunks:
                self.temp_file.write(chunk)

        if self.temp_file is not None:
            self.temp_file.write(text)

        self.chunks.append(text)
        self.chunks_bytes += len(text.encode("utf-8"))

        while self.chunks_bytes > self.rolling_bytes and len(self.chunks) > 1:
            removed = self.chunks.pop(0)
            self.chunks_bytes -= len(removed.encode("utf-8"))

    def current_tail_preview(self):
        return truncate_tail("".join(self.chunks), self.max_lines, self.max_bytes)

    def finalize(self):
        text = "".join(self.chunks)
        trunc = truncate_tail(text, self.max_lines, self.max_bytes)
        return trunc, getattr(self.temp_file, "name", None)
```

```python
def run_bash_tool(command, cwd, timeout=None, on_update=None, abort_signal=None,
                  env=None, command_prefix=None, spawn_hook=None, executor=None):
    resolved_command = command if not command_prefix else f"{command_prefix}\n{command}"
    context = {
        "command": resolved_command,
        "cwd": cwd,
        "env": build_env(env),
    }

    if spawn_hook is not None:
        context = spawn_hook(context)

    capture = OutputCapture()

    def on_data(raw_bytes):
        capture.add_bytes(raw_bytes)
        if on_update is not None:
            preview = capture.current_tail_preview()
            on_update({
                "content": preview.content or "",
                "details": {
                    "truncation": preview if preview.truncated else None,
                    "full_output_path": getattr(capture.temp_file, "name", None),
                },
            })

    exit_code = None
    cancelled = False

    try:
        exit_code = executor.exec(
            command=context["command"],
            cwd=context["cwd"],
            env=context["env"],
            timeout=timeout,
            abort_signal=abort_signal,
            on_data=on_data,
        )
    except TimeoutError:
        trunc, full_output_path = capture.finalize()
        output = trunc.content
        if output:
            output += "\n\n"
        output += f"Command timed out after {timeout} seconds"
        raise ToolFailure(output)
    except AbortError:
        trunc, full_output_path = capture.finalize()
        output = trunc.content
        if output:
            output += "\n\n"
        output += "Command aborted"
        raise ToolFailure(output)

    trunc, full_output_path = capture.finalize()
    output = trunc.content or "(no output)"

    if trunc.truncated:
        output += format_truncation_notice(trunc, full_output_path)

    if exit_code != 0:
        output += f"\n\nCommand exited with code {exit_code}"
        raise ToolFailure(output)

    return {
        "content": output,
        "details": {
            "truncation": trunc if trunc.truncated else None,
            "full_output_path": full_output_path,
        },
    }
```

## 16. Recommended defaults to copy

If you want behavior close to Pi, copy these defaults:

- shell-based execution, not manual argv parsing
- fixed tool cwd
- inherited env with optional overrides
- stdout + stderr merged into one stream
- streaming partial updates
- tail truncation for command output
- 2000 lines / 50 KB limits
- temp file for full output after threshold is crossed
- timeout support
- cancellation support
- kill full process tree
- non-zero exit treated as failure
- structured metadata for truncation/full-output path
- optional backend abstraction for remote execution
- optional context-exclusion flag for direct user shell commands

## 17. Main takeaways

The key design choices in Pi are:

1. **Commands are executed by a real shell.**
2. **Output is streamed live, but bounded by a rolling tail.**
3. **Large output is truncated aggressively and backed by a temp file.**
4. **Timeouts and cancellations kill the whole process tree.**
5. **Non-zero exit codes are surfaced as failures with full captured context.**
6. **Direct shell commands can be persisted as structured conversation events.**
7. **When the agent is already in the middle of a streamed turn, shell results are deferred to preserve message ordering.**

If you preserve those behaviors, your Python implementation will match Pi's design closely even if the libraries are completely different.
