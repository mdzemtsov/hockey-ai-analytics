# V10 night experiments

Read MORNING_REPORT.md for conclusions and QA_report.md for the full experiment history. No experimental tracker was promoted to V6.

## Reproduction dependencies

Run from the project root with the existing `/private/tmp/hockey-analytics-env/bin/python` environment; exact installed packages are in requirements_lock.txt. The results bundle is an evidence/code bundle, not a standalone application.

Required sibling files:

- V7 HockeyAI.pt detector and benchmark.py.
- V8 sam2.1_t.pt segmentation model.
- V5 botsort_reid.yaml.
- V9 source JPEGs/GT for four development sequences, cached tracks.csv, selections.json, run_match.py and its imports.
- V6 gate.py for the manual policy comparison.

The original V7/V8/V9 READMEs document model and source downloads. external_dependencies.json records hashes of directly used external local files. Original video/image data and model weights are excluded from the bundle. Source URLs and downloaded frame hashes are in long_holdout_sources.json and second_holdout_protocol_sources.json. Sources are third-party data with their original licensing; this archive does not grant additional rights.

Environment variables:

```sh
export YOLO_CONFIG_DIR="$PWD/hockey_ai_analytics/v7/runtime"
export MPLCONFIGDIR=/private/tmp/hockey-mpl-cache
```

Use the documented Python environment in place of `python` below.

## Experiments

Each `*_protocol.json` was written before its corresponding first execution. These are separate experiments with failures retained, not one cumulatively improved production tracker.

```sh
python hockey_ai_analytics/v10/audit.py
python hockey_ai_analytics/v10/run_appearance.py
python hockey_ai_analytics/v10/run_multisam.py
python hockey_ai_analytics/v10/replay_stable.py
python hockey_ai_analytics/v10/render_stable.py
python hockey_ai_analytics/v10/download_holdout.py
python hockey_ai_analytics/v10/run_long_baseline.py
python hockey_ai_analytics/v10/run_long_candidate.py
python hockey_ai_analytics/v10/review_long.py
python hockey_ai_analytics/v10/run_team_calibration.py
python hockey_ai_analytics/v10/run_multisam.py --deduplicate-seeds
python hockey_ai_analytics/v10/replay_stable.py --deduplicate-seeds
python hockey_ai_analytics/v10/replay_swap.py
python hockey_ai_analytics/v10/replay_swap.py --diagnostic-long
python hockey_ai_analytics/v10/test_manual_policy.py
python hockey_ai_analytics/v10/benchmark_trackers.py
python hockey_ai_analytics/v10/evaluate_tracker_selection.py
python hockey_ai_analytics/v10/download_holdout.py second_holdout_protocol.json
python hockey_ai_analytics/v10/second_holdout.py
python hockey_ai_analytics/v10/evaluate_second_selection.py
python hockey_ai_analytics/v10/render_second.py
python -m unittest discover -s hockey_ai_analytics/v10/tests -v
```

Some benchmark scripts skip runs with an existing metrics.json. To rerun, use a clean copy of the output folders while preserving the archived evidence. Do not mix cached output from different configurations.

## Review artifacts

- `stable/CHI_VS_TOR_004/comparison.mp4`: local V10-C improvement, 300 frames at30fps, **not independently validated**.
- `long_review/CAR_VS_NYR_001/comparison.mp4`: baseline vs rejected V10-C,1800 frames at60fps.
- `second_review/CNTRL_VS_PAC_001/comparison.mp4`: source/baseline/OC-SORT/DeepOCSORT,900 frames at30fps. Yellow and magenta indicate the two original bound IDs, not proof of correct identity.
- Each video's video_validation.json confirms full decoding/frame rate.
- Prediction and evaluation CSVs remain separate. UNKNOWN/abstention is counted as lost coverage, not a success.
- assistant_visual_review.json is a visual review of event contact sheets, not independent dense human ground truth.

No reference annotations were altered. Frame indices in research CSVs are1-based; V6 manual policy state CSVs explicitly use0-based frame_index. Native timestamps are preserved in rendered videos and manual policy logs.
