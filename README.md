# Hockey AI Analytics

**Partner: start with [the handoff guide](docs/PARTNER_START_HERE.md).**

Private, experimental local video review for hockey: explicit player selection, conservative tracking, pause-point graphics and English/Russian synthesized narration.

**Not a validated autonomous coach.** Team recognition and puck detection remain inconsistent. Player identities, possession, zones, roles, good/bad decisions and safe skating routes are not established. Read [the baseline QA](docs/BASELINE_QA.md) before interpreting output.

## Run locally

Use Python 3.13. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r hockey_ai_analytics/v6/requirements.txt
python scripts/check_assets.py
python hockey_ai_analytics/v6/server.py --port 8765
```

Open `http://127.0.0.1:8765/coach` for coaching review or `/` for manual player selection. The server binds only to localhost. Upload a MOV/MP4, choose an interval of up to 30 seconds for coaching, and select English or Russian. macOS uses installed `say` voices; other systems use browser speech where available. No audio is sent to a remote speech provider.

Model files are deliberately excluded. See [assets.json](assets.json) for locations, upstream sources and SHA256 hashes. Acquire assets separately and check upstream licensing. Upload your own authorized footage. The original private demo and cached detections are not included, so the demo button is hidden on a clean checkout.

Existing setup was tested on macOS/Apple Silicon. Installation on other platforms and a completely new dependency environment is not yet verified. The pinned application requirements preserve the current prototype; machine-specific runtime details are in the original local research archive.

## Test without footage or models

```sh
python -m pip install -r requirements-test.txt
python -m unittest discover -s hockey_ai_analytics/v6/tests -p 'test_*.py' -v
```

These tests cover selection safety, cache integrity and coaching rules using synthetic inputs. They do not measure real-world recognition accuracy. Private video replay tests remain in the original local workspace and are excluded from this repository.

## Layout

- `hockey_ai_analytics/v6/`: active app and tests.
- `hockey_ai_analytics/v5/botsort_reid.yaml`: tracking configuration used by the app.
- `hockey_ai_analytics/v5`–`v15/`: earlier trackers, experiments, training/evaluation code, protocols and reports.
- `experiments/v15/`: initial standalone rejected color-candidate snapshot; the fuller historical package is under `hockey_ai_analytics/v15/`.
- `docs/`: QA baseline, data policy and research roadmap.
- `assets.json`: externally obtained model manifest.

The versioned directory layout is retained so the working app's relative paths remain valid. Generated files go under ignored `hockey_ai_analytics/v6/work/`. Reopening a review URL requires the same server session; saved files persist, but job registration does not yet survive restart.

## Development rules

Keep source media, player crops, raw annotations, credentials and model binaries outside Git. Never promote a model because it produces more labels: compare errors and abstentions on a fixed validation set. No future ground truth in inference. Explicit human selection is not independent ground truth. Do not identify the test player as Matvey. See [data policy](docs/DATA_POLICY.md).

This source handoff now includes the active app plus first-party historical research across V4–V15, training/evaluation code, protocols, aggregate metrics and QA. It is not a complete data/checkpoint archive. Existing local videos and historical outputs remain untouched. No open-source license is granted by this private repository; third-party code and models retain their own terms.
