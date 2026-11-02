# Key bindings

Mother's keys can all be remapped in `~/.config/mother/config.toml` with a
`[keybindings]` table. The table maps a stable binding id to a Textual-style key
string:

```toml
[keybindings]
submit = "ctrl+enter"
newline = "enter"
toggle_auto_scroll = "ctrl+t"
markdown_cursor_down = "n"
markdown_cursor_up = "p"
```

Values are case-insensitive and are normalized to lowercase. Key names use
Textual syntax, such as `enter`, `escape`, `ctrl+n`, `shift+g`, or `ctrl+shift+g`.
A value may contain a comma-separated list to bind more than one key to the same
action, for example `toggle_auto_scroll = "ctrl+g,f2"`.

Unknown binding ids are rejected with an error that lists the valid ids, so
typos do not silently do nothing.

## Defaults

### Prompt input

| Binding id | Default | Action |
| --- | --- | --- |
| `submit` | `enter` | Submit the prompt (also submits a built-in slash command or accepts a completion when the popup is open) |
| `newline` | `shift+enter` | Insert a newline / accept the open completion popup |
| `history_search` | `ctrl+r` | Open fuzzy prompt-history search |
| `cycle_prompt_size` | `ctrl+e` | Cycle the prompt between normal, half-screen, and full-screen |

### App-wide actions

| Binding id | Default | Action |
| --- | --- | --- |
| `toggle_thinking_widget` | `ctrl+o` | Expand or collapse the focused output / the latest collapsible output |
| `toggle_auto_scroll` | `ctrl+g` | Toggle automatic scrolling |
| `show_models` | `ctrl+n` | Open the model picker |
| `scroll_to_bottom` | `end` | Jump to the bottom of the conversation |
| `scroll_to_bottom_from_chat` | `shift+g` | Jump to the bottom while reading chat blocks |
| `save_session` | `ctrl+s` | Save the session as Markdown |

`ctrl+p` (the command palette), `ctrl+q` (quit), and other built-in Textual
bindings are not part of Mother's keybinding table.

### Output and Markdown blocks

| Binding id | Default | Action |
| --- | --- | --- |
| `copy_output` | `c` | Copy the focused output block |
| `toggle_expanded` | `ctrl+o` | Expand or collapse the focused output block |
| `markdown_cursor_down` | `j` | Select the next Markdown block |
| `markdown_cursor_up` | `k` | Select the previous Markdown block |
| `markdown_copy_block` | `c` | Copy the selected Markdown block |

### Model picker

| Binding id | Default | Action |
| --- | --- | --- |
| `model_picker_dismiss` | `escape` | Close the model picker |

### Session picker

| Binding id | Default | Action |
| --- | --- | --- |
| `session_picker_dismiss` | `escape` | Close the session picker |
| `session_picker_delete` | `delete` | Delete the selected session |
| `session_picker_rename` | `r` | Rename the selected session |

### Bash approval dialog

| Binding id | Default | Action |
| --- | --- | --- |
| `bash_approval_approve` | `y` | Run the reviewed command |
| `bash_approval_approve_enter` | `enter` | Run the reviewed command |
| `bash_approval_deny` | `n` | Deny the command and copy it to the clipboard |
| `bash_approval_deny_escape` | `escape` | Deny the command and copy it to the clipboard |

## Legacy prompt keys

The older top-level fields are still supported:

```toml
submit_key = "ctrl+enter"
newline_key = "enter"
```

`keybindings.submit` and `keybindings.newline` take precedence over these fields
when both are present. `submit` and `newline` must always resolve to different
keys.

Whether combinations such as `shift+enter` or `ctrl+enter` arrive in the app
depends on what your terminal sends. If a binding seems inert, try a key your
terminal definitely passes through.
