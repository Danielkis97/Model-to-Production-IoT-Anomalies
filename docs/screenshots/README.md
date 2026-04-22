# Terminal Screenshots

This folder holds four screenshots that are embedded in the main README:

| File | Capture command |
|---|---|
| `training.png` | `python -m app.train_eval` |
| `docker_build.png` | `docker compose up --build` (during image build + container launch) |
| `docker_logs.png` | `docker compose up --build` (live API + sender stream logs) |
| `tests.png`    | `python tests/test.py` |

## How to capture on Windows

1. Run the command in a fresh terminal window
2. Wait for the output to finish
3. Press `Win + Shift + S` → rectangular snip → drag across the terminal
4. Paste into Paint (`Win + R` → `mspaint`), then `File → Save As` → PNG
5. Save as `docs/screenshots/<name>.png`

## How to capture on macOS

1. `Cmd + Shift + 4` → drag across the terminal → PNG lands on Desktop
2. Rename and move to `docs/screenshots/<name>.png`
