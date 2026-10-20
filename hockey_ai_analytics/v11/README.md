# V11 — identity-recovery development benchmark

Status: recovery v1 implemented and tested. Development gains did not generalize sufficiently to the reserved continuation; candidate rejected for deployment. See `RECOVERY_QA.md`.

## What was completed

Seven difficult selected-player cases from Chicago–Toronto, grouped into four events: two player crossings, an exit/re-entry window, and a fragmented trajectory near the boards. Saved source videos, dense publisher references, seed-only selection requests, sampled reference contact sheets, and per-frame evaluations of three existing methods.

There are 357 case-frame observations over 199 unique source images. Paired cases share images and must not be treated as independent samples or split across development and validation.

| Case | Baseline correct/visible | OC-SORT | DeepOCSORT | Wrong-person frames, baseline / OC / Deep |
|---|---:|---:|---:|---:|
| cross_blue | 14/41 | 38/41 | 13/41 | 23 / 0 / 23 |
| cross_white | 16/41 | 39/41 | 16/41 | 25 / 0 / 25 |
| boards_blue | 15/46 | 13/46 | 13/46 | 20 / 0 / 0 |
| boards_white | 26/46 | 46/46 | 26/46 | 20 / 0 / 20 |
| edge_skater | 8/32 | 7/32 | 7/32 | 0 / 0 / 0 |
| edge_goalie | 6/31 | 0/31* | 0/31* | 0 / 0 / 0 |
| near_boards | 13/41 | 22/41 | 9/41 | 0 / 0 / 0 |

*Initial seed binding failed. These are not successful selections followed by recovery failures.

OC-SORT avoids the measured wrong-player outputs in these cases, but target loss remains severe. This deliberately selected failure set cannot establish general superiority; V10's wider comparisons found regressions on other clips.

## Reference and evaluation limitations

- References are unchanged publisher personnel labels, independent of the evaluated trackers. They are NOT newly independently human-verified ground truth. `review_queue.csv` remains pending. An assistant sampled visual inspection of `cross_blue` is not a dense annotation audit.
- Published annotation absence is treated as absence for scoring; annotation omissions remain a possible source of error.
- Seed boxes use the first-frame publisher box as an ideal manual-selection surrogate. No future reference is used to select identities. This does not test real user click/box accuracy.
- Existing causal tracker outputs include the original clip prefix. The short cases are not fresh tracker restarts. Predictions are saved before future reference is read for scoring.
- Correct means IoU >= 0.5 to the selected reference. Wrong-person frames require overlap >= 0.5 with another reference person when not correct for the target. They are not ID-switch counts.
- All frame rows are retained, including misses and absence. Low wrong-person counts alone do not establish success.
- Recovery delay describes only the first failure followed by three correct frames. Later failures can occur, so a short delay does not imply sustained recovery. Read the per-frame CSV and coverage alongside it.
- No human confirmations were used. No tactical, puck, team-role or Matvey-identification claims follow from this benchmark.

## Reproduction

From the project root, using the existing local Python environment:

```sh
/private/tmp/hockey-analytics-env/bin/python hockey_ai_analytics/v11/build_benchmark.py
/private/tmp/hockey-analytics-env/bin/python hockey_ai_analytics/v11/benchmark_baselines.py
/private/tmp/hockey-analytics-env/bin/python -m unittest discover -s hockey_ai_analytics/v11/tests
/private/tmp/hockey-analytics-env/bin/python hockey_ai_analytics/v11/report.py
```

Dependencies: V9 Chicago–Toronto source JPEGs and publisher GT, V9 baseline tracks, V10 OC-SORT/DeepOCSORT tracks. Source image hashes are in `source_manifest.json`; reference provenance is in `reference_provenance.json`. This folder is not a standalone model distribution. `artifact_manifest.json` covers saved V11 evidence.

Open `review.html` to inspect source clips, sampled references and comparison tables.

## Implemented recovery experiment

`recovery.py` adds immutable first-frame appearance, motion gating, a common-motion camera proxy, overlap/ambiguity abstention, three-frame reacquisition confirmation and expiry requiring explicit reselection. The histogram descriptor is reused from V10; it is not a learned identity or jersey recognition model.

Run `run_recovery.py` for development replay, `validate_reserved.py` for the locked reserved comparison, and `render_recovery.py` for comparison videos and coordinate/state CSVs. `candidate_lock.json` records candidate/configuration hashes before the reserved download. All 13 unit tests pass. `reserved/summary.json` contains the completed reserved results.

The 901–1500 interval has now been consumed for validation and must not be called unseen in later experiments. A future candidate needs another untouched interval or match.

## Next implementation checkpoint

Build a conservative identity-recovery layer that links short trajectories using motion, camera movement, appearance and team evidence. Preserve UNKNOWN when multiple candidates are plausible; a tracker ID alone is not identity evidence. Begin with selected-player mode. Evaluate failed initial selection separately from reacquisition.

The completed test followed the pre-download lock rule on CNTRL_VS_PAC_001 frames 901–1500 (20 seconds). This is a temporal continuation of a previously studied match, not an independent-match holdout. Keep publisher GT outside inference; do not tune on these results and call the same interval validation again. Subsequent independent matches and independent annotation review remain required.

Do not restart the overnight automation. V10 evidence and archive remain frozen; V6 application is unchanged.
