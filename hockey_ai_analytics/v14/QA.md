# Hockey-specific identity training — completed, not qualified for deployment

20 October 2026. Two training experiments were completed: the small V13 pilot and the expanded V14 run. Both weights and all results are preserved. Neither model is connected to the V6 application.

## What was actually trained

A 131,456-parameter metric head over frozen DINOv2 image features: 384 → 256 → 128 dimensions, normalized output. This is real hockey-specific supervised training, not another unmodified general model, but it is not full-backbone fine-tuning.

The loss combines supervised contrastive learning with a same-team hard-negative triplet term. Each batch contains four real views per training identity; same-team players must be separated rather than merely assigned the right uniform colour. No automatic recovery predictions are used as positive training labels.

| Run | Training tracks | Training crops | Held-out calibration tracks | Calibration crops | Footage span per training match |
|---|---:|---:|---:|---:|---|
| V13 | 16 | 342 | 6 | 142 | first 300 frames / 5 seconds |
| V14 | 28 | 1,413 | 6 | 346 | first 3,000 frames / 50 seconds |

Both use three training matches, a fixed seed, 100 epochs and four batches per epoch. V14 uses 600 temporally sampled training images. Its eight Chicago–Toronto enrollment identities have 89 real views in total, at most 12 per identity, taken only from frames 1–300.

## Review and label quality

Training references come from unchanged [VIP-HTD personnel annotations](https://github.com/harshap-ai/VIP-HTD), which are intended to preserve identity across exits. Assistant review inspected four temporal crops per candidate and excluded referees, goalies and uncertain identities.

The longer audit exposed label inconsistencies: CAR–BOS ID 7 spans visibly different goalie teams; STL–SJ ID 11 changes team appearance; STL–SJ ID 6 includes goalie/skater inconsistency. CGY–DAL ID 33 was excluded as a possible duplicate of ID 4, both showing Dallas jersey 18 in sampled views. These are recorded exclusions, not silent changes to publisher GT.

**Independent human confirmation of every view has not happened.** The new `review_labels.html` displays every one of the 1,759 training/calibration crops and exports a separate review overlay. It starts with all identities pending. Nothing is auto-approved, and exporting a review does not silently change source annotations or retrain the model.

Match/team splits and publisher track IDs are separate. Historical roster identities have not been verified across matches, so do not claim proven real-person disjointness. See DATA_LIMITATIONS.md.

## New V14 test

Model, training data, enrollment gallery, thresholds and code were frozen before downloading Chicago–Toronto frames 901–1500. This is a new temporal interval from a previously studied match, not an entirely untouched match.

Scoring samples every 15th frame: **40 query images**, producing 157 enrolled-player queries and 162 unenrolled-personnel queries. All 600 source frames are saved, but this is not a dense 600-frame tracking test.

Each query crop uses a publisher box. The model chooses an identity from the earlier labelled gallery; future identity labels are used only for scoring. This oracle-crop test isolates identity matching and does not prove automatic detection or tracking.

| Representation | Maximum enrollment views | Correct top-1 / 157 known queries | Accuracy | Same-team errors |
|---|---:|---:|---:|---:|
| Original DINOv2 | 1 | 50 | 31.8% | 85 |
| Original DINOv2 | 4 | 56 | 35.7% | 72 |
| Original DINOv2 | 12 | 61 | 38.9% | 77 |
| Hockey-trained head | 1 | 35 | 22.3% | 86 |
| Hockey-trained head | 4 | 47 | 29.9% | 81 |
| Hockey-trained head | 12 | 40 | 25.5% | 98 |

The wider original gallery improves matching on this interval. The trained head underperforms the original descriptor at every tested gallery size, including more same-team errors with 12 views.

The predeclared calibration rule sets a top-two identity margin above all observed calibration mistakes, with a 0.03 floor. **All six variants abstain on every test query at their frozen calibrated thresholds.** This is zero useful accepted coverage, not perfect recognition. No threshold was loosened after inspecting the test.

The earlier V13 experiment had 41/209 top-1 correct matches for its trained four-view model versus 37/209 for the generic comparator, but increased same-team errors and accepted no trained matches. V13 and V14 use different query intervals, so their raw accuracies are not directly comparable.

## Decision

Reject both trained models for automatic identity recovery. This implementation completed the training experiment but did not solve reliable player recognition. Fitting the training identities and reducing the training loss did not generalize adequately.

The evidence points to insufficiently verified identity labels, limited data diversity and single-crop appearance ambiguity as remaining problems. It does not establish which factor is dominant. Changing only acceptance thresholds would trade abstention for false reconnections.

## Reproducibility

- Model: `hockey_metric_head.pt`; V13 model remains in `../v13/hockey_metric_head.pt`.
- Frozen source/config/data/model hashes: `candidate_lock.json`.
- Dataset manifests: `features.json`, `features_sources.json`, `gallery.json`, `gallery_sources.json`.
- Sampling and exclusion policy: `protocol.json`, `audit_labels.json`, `gallery_labels.json`.
- Training record: `training_history.json`, `training_summary.json`.
- Calibration: `calibration.json`, `calibration_*.csv`, `calibration_split.json`.
- Test: `test/summary.json`, six per-query CSVs, source manifest and extracted feature file.
- Review: `review.html`, `review_labels.html`, `review_crops/`.
- Six tests pass per experiment for metric shape/normalization, finite training gradients, same-team-negative handling, gallery identity aggregation, unknown-person accounting and split boundaries.

The original `sources/`, V10 evidence archive, and V6 application are unchanged. Test players are not identified as Matvey. No automatic overnight runs were resumed.
