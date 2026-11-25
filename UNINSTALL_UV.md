# Uninstalling uv (used to launch the Blender MCP)

uv (Astral) was installed only so the Blender MCP server could start through
`uvx blender-mcp`, which is configured in `~/.claude.json` under this project's
`mcpServers.blender`. This file says how to remove uv completely.

**To Claude:** when asked to uninstall uv, follow these steps in order. Run step 1
*before* step 2, because step 1 needs `uv` to still exist. Show the output of step 3
to the user. Confirm with the user before step 4, because it changes the MCP config.

## What uv leaves on the machine

| What | Where (Windows) | Created by |
|---|---|---|
| `uv.exe`, `uvx.exe` | winget package folder, plus a PATH entry or link | `winget install` |
| Package cache (blender-mcp and its dependencies) | `%LOCALAPPDATA%\uv\cache` | every `uvx` run |
| Managed Python builds (only if uv downloaded one) | `uv python dir`, under `%APPDATA%\uv` | `uvx`, when it wants its own Python |
| Installed tools (only if `uv tool install` was ever used) | `uv tool dir`, under `%APPDATA%\uv` | `uv tool install` |

uv doesn't install a service, doesn't create other registry entries, and doesn't touch
the system Python (`C:\Python313`).

## Steps

### 1. Clear uv's data (uv must still be installed)

```powershell
uv cache clean
```
```powershell
Remove-Item -Recurse -Force "$(uv python dir)"
```
```powershell
Remove-Item -Recurse -Force "$(uv tool dir)"
```

If one of these folders doesn't exist, the error is harmless. Continue.

### 2. Remove the program

```powershell
winget uninstall --id=astral-sh.uv -e
```

If uv was installed some other way (not winget), delete `uv.exe`, `uvx.exe` and
`uvw.exe` from `%USERPROFILE%\.local\bin` instead.

### 3. Verify it's gone

Open a **new** PowerShell window so PATH is reloaded, then run:

```powershell
Get-Command uv, uvx -ErrorAction SilentlyContinue; Test-Path "$env:LOCALAPPDATA\uv", "$env:APPDATA\uv"
```

Expected result: no commands are listed, and it prints `False` twice. If either path
prints `True`, delete that folder:

```powershell
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\uv", "$env:APPDATA\uv" -ErrorAction SilentlyContinue
```

### 4. The Blender MCP entry (optional)

Without uv, the `blender` MCP server in `~/.claude.json` fails with
"Connection closed" again. Either:

- remove the `"blender": { ... }` block from this project's `mcpServers` in
  `~/.claude.json`, or
- switch it to a self-contained venv instead, which needs no uv. Create a venv (for
  example `C:\tools\blender-mcp`), run `pip install blender-mcp` inside it, and set
  `"command"` to that venv's `Scripts\blender-mcp.exe` with `"args": []`. To remove
  that later, delete the folder and the config block.

Restart the Claude desktop app after any change to `~/.claude.json`.

## Not part of uv (leave alone)

- The Blender MCP **addon** inside Blender (it listens on `127.0.0.1:9876`). Remove it
  in Blender: Edit > Preferences > Add-ons.
- `armor-forge/.venv`, the project's own test environment, made with the system
  `python -m venv`, not uv.
