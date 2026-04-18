# UI Workflow

Use this file when you want to run the local image-upload UI instead of calling
`main.py` manually.

## Goal

Keep the heavy work on the laptop:

- choose an image from the computer
- preview the extracted edges or ML mask
- preview the ordered drawing paths
- compare AI variants on the same uploaded image
- generate `plot_commands.txt`
- optionally send commands to the live bridge
- optionally export a CyberPi `.py` file and `.mcode`

## Start The UI

From the project root:

```powershell
python -m src.ui_server
```

Optional:

```powershell
python -m src.ui_server --open-browser
```

Default address:

- `http://127.0.0.1:8010`

## Main Flow

The UI is intentionally minimal and step-based:

- Step 1: choose an image
- Step 2 appears only after an image is selected
- Step 3 and Step 4 appear only after generation succeeds
- on laptop/desktop, the status rail stays sticky on the left
- the left status rail can be closed and opened
- the page intentionally uses most of the laptop width instead of keeping large side margins
- advanced settings stay collapsed by default
- robot send and export remain secondary, optional actions

1. Open the browser at `http://127.0.0.1:8010`
2. Choose an image file
3. Click `Generate Preview And Commands`
4. Check:
   - original preview
   - detected edges / mask
   - planned path preview
   - generated command count
5. In Step 4, decide whether you want comparison first or testing first:
   - `AI Experiment Lab` to compare classical / ML and planning variants
   - `Manual mBlock Paste Mode` for simple testing first
   - `Automatic Live Bridge Mode` for direct send testing
6. For AI comparison:
   - keep `classical` checked at minimum
   - optionally add `ml` if you have a checkpoint
   - compare `nearest_neighbor` and `two_opt`
   - optionally upload a ground-truth mask for Dice / IoU
   - click `Run AI Comparison`
   - review the per-run cards and the summary box
7. For manual mBlock testing:
   - use `Prepare Manual Test Files (.py + .mcode)`
   - then open the Python script
   - or show/copy the Python script from the UI and paste it into mBlock
8. For automatic testing:
   - use `Send Through Live Bridge`

## Output Location

Each UI generation creates a timestamped folder in:

- `output/ui_sessions/`

Typical files inside one session:

- uploaded image copy
- `edges_preview.png`
- `path_preview.png`
- `plot_commands.txt`
- `paths_summary.json`
- `ui_request.json`
- `ui_result.json`
- `exports/*.py`
- `exports/*.mcode`
- `experiments/<timestamp>/ui_compare_result.json`
- `experiments/<timestamp>/<variant>/edges_preview.png`
- `experiments/<timestamp>/<variant>/path_preview.png`

## Verified Smoke Test

The UI was smoke-tested locally with `images/Testlogo.jpeg`.

Verified:

- upload and processing worked
- generated output had `741` commands and `26` paths
- `.mcode` export worked
- UI send action worked in file mode
- AI comparison worked for:
  - `classical + nearest_neighbor`
  - `classical + two_opt`
- the comparison summary reported the expected `9.23%` planning improvement

Smoke-test artifacts include:

- `output/ui_sessions/20260417_120438_508860/`
- `output/ui_send_smoke.jsonl`
- `output/ui_sessions/20260418_012418_239022/experiments/20260418_012418_567999/`

## Notes

- `Send Latest Commands` in `socket` mode expects the mBlock live socket
  receiver to already be running.
- `Send Latest Commands` in `file` mode writes JSONL to the configured outbox.
- `ml` vision mode still requires a trained checkpoint path.
- After export, the UI can open the generated Python script directly for
  copy/paste testing in mBlock.
- The UI intentionally recommends manual paste mode before the automatic bridge
  mode because it is easier to debug.
- The AI Experiment Lab is the easiest way to show the course-oriented AI story
  without leaving the browser UI.
