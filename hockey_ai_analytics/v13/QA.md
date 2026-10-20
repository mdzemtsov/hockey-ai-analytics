# V13 pilot — trained metric head, not ready

Trained 131,456 projection-head parameters over frozen DINOv2 features using 342 crops from 16 publisher track identities across three hockey matches. Six different track identities (142 crops) were withheld from training. The objective includes supervised contrastive loss and an explicit same-team hard-negative triplet term. Schedule: 100 epochs, four batches per epoch, fixed seed 137.

Training views are actual crops, not interpolation. An assistant reviewed four sampled views per identity, recorded team labels and excluded referees, goalies and uncertain examples. Independent human verification remains pending; see DATA_LIMITATIONS.md.

## Frozen test

Chicago–Toronto enrollment: eight publisher identities, four actual labelled views each from frames 1–120. Query interval: newly downloaded frames 301–900, with every 15th frame scored. This produces 209 enrolled-player crop queries and 116 unenrolled-personnel queries from 40 images. Publisher boxes give an oracle-crop advantage; this is not a tracking test.

| Representation / gallery | Correct top-1 / 209 known queries | Same-team errors | Accepted known | Accepted unknown | Correct / all accepted |
|---|---:|---:|---:|---:|---:|
| Generic / one view | 31 | 105 | 0 | 1 | 0/1 |
| Generic / four views | 37 | 97 | 7 | 4 | 3/11 |
| Trained / one view | 34 | 111 | 0 | 0 | 0/0 |
| Trained / four views | 41 | 120 | 0 | 0 | 0/0 |

The trained four-view model improves top-1 from 17.7% to 19.6%, but same-team confusion rises and its calibrated gate rejects every query. No automatic reconnection or deployment is justified.

Thresholds were calibrated only on withheld training-match identities, then frozen with model weights before downloading query images. Calibration precision is a fitted statistic, not a guarantee on new footage. Test results were not used to loosen thresholds.

Six tests pass for normalized embeddings, gradients, same-team-negative handling, per-identity gallery aggregation, unknown-person errors and dataset split boundaries. Reproduction: prepare_features.py, train.py, retrieval.py, download_test.py, test_holdout.py, in that order with the existing Python environment. Preserve candidate_lock.json rather than overwriting it.

Model: hockey_metric_head.pt. Per-query results: test/*.csv. Training and calibration manifests, source hashes, model history and reference provenance are saved alongside them.

Decision: retain as failed small-data baseline. V14 expands training-view diversity and enrollment using different future test frames; V13's consumed query interval must not be called unseen again.
