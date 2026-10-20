# Team-color audit and coaching safeguards — 20 October 2026

## Outcome

A broader jersey-color candidate was tested and **rejected**. It labels more crops but increases wrong assignments. Only coaching safeguards were integrated (coach-evidence-v1.2); the color classifier is unchanged.

## Fixed crop audit

Deterministic first and last sampled frame from each of the existing original, Chicago–Toronto and Carolina–Boston reviews. 51 detected-player crops in total. First samples are development; last samples are validation. Labels were assigned by the assistant from rendered contact sheets, not independent human ground truth. No real-player identity annotations or Matvey identification are implied.

Labels represent visible jersey families: red, white, blue and Boston gold/yellow (which includes black panels). Crops containing substantial parts of multiple people are MIXED; visible referee detections are REFEREE. These are reported separately rather than silently counted as correctly colored players. Contact sheets and all labels are retained for correction by a reviewer. Crops share matches across development and validation; this is not a held-out-match generalization benchmark.

The candidate broadened the torso sample and allowed chromatic evidence despite central numbers/stripes. Its code hash was frozen in candidate_lock.json before validation images were viewed. There was no adjustment after validation. The original model detections were held fixed; detector misses outside these crops are not measured.

| Set | Method | Eligible crops | Correct | Wrong | UNKNOWN | Mixed/referee assigned a color |
|---|---|---:|---:|---:|---:|---:|
| Development | Original | 20 | 9 | 2 | 9 | 1/4 |
| Development | Candidate | 20 | 13 | 5 | 2 | 3/4 |
| Validation | Original | 24 | 7 | 3 | 14 | 1/3 |
| Validation | Candidate | 24 | 12 | 5 | 7 | 3/3 |

The increase from 7 to 12 correct validation labels does not justify deployment: wrong assignments rise from 3 to 5, and mixed-person crops are classified more often. These descriptive counts are subject to assistant-label error; no calibrated confidence or production accuracy claim.

## Implemented safeguards

1. Unknown-color players, goalies and referees now block candidate passing lanes. Previously only known opposing-color players were checked, treating other detected people as empty space.
2. An explicitly requested team color is never silently replaced with the apparent puck-nearest player's color.
3. A third known color makes the two-team hypothesis unresolved even if only one person has that color. It is not silently discarded.

These changes reduce unsupported suggestions; they do not improve recognition or prove that an unblocked image line is a safe pass. Updated narration says no other detected person occupies the line, preserving the caveat about unseen people, sticks and perspective.

27 unit/replay tests pass, including the new lane-blocker, requested-team and third-color regressions. Existing hockey model, inference thresholds and player identity safeguards remain unchanged. Automated service test evidence is in ../v6/tests/coach_service_result.json. Updated local service port: 8770.

## Reproduction and artifacts

- `audit.py`: regenerate contact sheets and deterministic crop manifest from original saved results.
- `labels.json`: assistant visual labels; not silently altered after evaluation.
- `color_candidate.py`, `candidate_lock.json`: rejected classifier and frozen fingerprint.
- `evaluate.py`: compare original and candidate against fixed labels.
- `comparison.json`, `summary.json`: per-crop outputs and counts.
- `development.jpg`, `validation.jpg`: visual audit sheets.

Run with the existing /private/tmp/hockey-analytics-env/bin/python runtime. Nothing under sources/ or original job outputs was edited. No raw media or coordinates uploaded to Notion. Automation remains paused.

Final service verification passed in both team and genuinely bound selected-player modes on port 8770, with 10 generated/served narration files per mode across English and Russian. No tactical arrows were justified. Results are preserved in service_result.json; team review job 6dd5d4faa1c749cdb0d80ae1c907feda.
