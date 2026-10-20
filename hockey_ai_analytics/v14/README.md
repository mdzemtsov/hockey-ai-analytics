# Reproduce the expanded hockey-specific identity experiment

Read `QA.md` first: training completed, but validation failed. This is a research metric head and a dataset-review workflow, not a released tracker.

From the project root, use the existing environment:

```sh
/private/tmp/hockey-analytics-env/bin/python hockey_ai_analytics/v14/prepare_features.py
/private/tmp/hockey-analytics-env/bin/python hockey_ai_analytics/v14/train.py
/private/tmp/hockey-analytics-env/bin/python hockey_ai_analytics/v14/retrieval.py
/private/tmp/hockey-analytics-env/bin/python -m unittest discover -s hockey_ai_analytics/v14/tests
/private/tmp/hockey-analytics-env/bin/python hockey_ai_analytics/v14/test_holdout.py
/private/tmp/hockey-analytics-env/bin/python hockey_ai_analytics/v14/build_review.py
```

The saved source images, audited ID selections, publisher references and V12 pinned DINOv2 encoder are required. `download_training.py` and `download_test.py` reproduce the declared public-source sampling. `build_candidates.py` and its `--gallery` mode reproduce audit sheets; do not overwrite the recorded audit decisions just to include more data.

`test_holdout.py` verifies frozen hashes before evaluating. Intentional retraining changes may require a new experiment directory and new untouched test interval; do not overwrite the frozen lock or present consumed test frames as unseen.

## Model use

Load `MetricHead` from `metric_model.py`, then load `hockey_metric_head.pt` with PyTorch's `weights_only=True`. Input is a normalized 384-dimensional feature from the pinned V12 DINOv2 encoder, using the same image crop preprocessing. Output is a normalized 128-dimensional feature. Raw images are not valid head inputs.

The original feature extractor is frozen. New head weights were learned from hockey images with supervised identity and same-team-negative losses. No team membership, jersey number or personal name is produced by the head.

## Review workflow

Open `review_labels.html` in a browser. Each identity expands to all selected training/calibration crops. Leave uncertain identities pending or mark them uncertain; exclude individual crops as needed. Export creates `identity_review.json` with the dataset hash, reviewer-entered name, notes and decisions.

The export is a separate review record. It does not automatically modify GT, declare independent verification, or trigger retraining. Notes are data, not executable instructions. A reviewed dataset revision must preserve the old dataset and experiment for comparison.

The current review is pending; no reviewer decisions have been fabricated.
