# Setup and external assets

## Active app

1. Clone this public repository; no invitation is required to read it.
2. Create a Python 3.13 virtual environment in the repository root.
3. Install `hockey_ai_analytics/v6/requirements.txt`.
4. Acquire the two weights in `assets.json` from their named upstream sources, respecting their licenses, and place them at the declared paths. `python scripts/fetch_assets.py --download` performs a hash-checked download; it does not install or execute model code.
5. Run `python scripts/check_assets.py` and resolve missing/hash-mismatched files.
6. Run `python hockey_ai_analytics/v6/server.py --port 8765`.
7. Open `http://127.0.0.1:8765/coach` and upload a short authorized clip.

The original macOS launcher is historical and prefers a machine-local temporary environment. For a partner's machine, use the explicit virtual-environment commands in the root README.

## Test without assets

`python -m pip install -r requirements-test.txt` followed by the README's unittest command. GitHub CI runs these core tests on Linux. This does not verify a full model installation or inference on a partner's machine.

## Speech

macOS uses installed English/Russian `say` voices to create local M4A audio. On other systems the browser may provide synthesis. If the requested voice is missing, the UI reports it and retains the transcript. No paid voice service or API key is required. Pronunciation has not been independently rated.

## Research assets

Public dataset download scripts are provided in V7, V9, V10, V13 and V14. They can be large and require network access. Their sources and protocols describe the intervals. Do not download everything merely to start the active app.

DINOv2, SAM2, OCR/number-recognition repositories and research weights are not vendored. See version-specific source/dependency manifests and third-party notices. The V13/V14 trained heads are failed research artifacts and deliberately not included as app assets.

## Troubleshooting

- Missing weights: check assets.json and run check_assets.py. Do not rename an arbitrary model to satisfy the path.
- Demo button absent: expected without the private source video.
- Old job URL fails after restart: job registration is not persisted yet; use a new upload. Generated files remain in `v6/work/`.
- UNKNOWN everywhere: inspect detection and team/puck evidence; never weaken thresholds simply to make the demo look active.
- Package installation fails: preserve the error and inspect platform/Python compatibility. Historical requirements_lock files record past environments, not universally supported installation recipes.
