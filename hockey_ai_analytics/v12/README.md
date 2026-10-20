# V12 — explicit team and learned visual-identity evidence

Experimental selected-player engine; not enabled in V6. Read `QA.md` for the measured result before using this candidate.

## What changed

Three variants share exactly the same detector outputs, original V11 motion/appearance recovery code, selection and scoring:

1. `v11`: unchanged recovery baseline.
2. `team_only`: additionally requires agreement with two explicitly selected opposing-team examples. Central upper-jersey histograms avoid much of the pants/background included in older appearance features. Uncertain classification rejects the automatic observation.
3. `team_identity`: additionally compares frozen DINOv2 crop embeddings against the selected first-frame player and other first-frame people. Both positive similarity and separation from competing identities are required.

The immutable first-frame templates avoid learning an incorrectly reacquired person. These templates are also vulnerable to pose, scale, lighting and camera changes. No jersey number or personal identity is inferred. The two-team check is an appearance hypothesis, not independently measured team recognition accuracy.

Every automatic observation is checked, even if its tracker ID stays unchanged. Rejected candidates remain available to the original camera-motion and overlap checks; removing them would hide occlusions. Initial manual selection is authoritative. No future positions, GT IDs, or interpolated labels are available to the inference classes.

## Model provenance

The learned descriptor is [Meta's official DINOv2 ViT-S/14](https://github.com/facebookresearch/dinov2). It is a general image-feature model, **not a hockey-specific player re-identification model**. It has not been fine-tuned here.

`encoder_source.json` pins the upstream source commit and downloaded weight SHA256. `candidate_lock.json` records source/config/model hashes before validation images were downloaded. Upstream code/model files remain in `hub/`; they are excluded from the general artifact manifest but their provenance and weights are recorded.

## Evaluation scope

- Development: the consumed V11 CNTRL–PAC interval, frames 901–1500, 20 seconds, two original first-frame visual selections.
- Validation: newly downloaded CAR–NYR frames 1801–2400, 10 seconds at 60 fps, two first-frame visual selections. This is a temporal continuation of another previously studied match, not independent-match validation.
- Existing publisher personnel GT is unmodified and read only after all predictions are saved. Separate human verification remains pending.
- There are 1,200 unique images across the two intervals and 2,400 target-frame observations across four selections. Variant repetitions do not create additional samples.
- Compare coverage and false identity together. UNKNOWN is not success; zero wrong-player frames alone cannot justify deployment.

## Reproduction

From the project root:

```sh
/private/tmp/hockey-analytics-env/bin/python -m unittest discover -s hockey_ai_analytics/v12/tests
/private/tmp/hockey-analytics-env/bin/python hockey_ai_analytics/v12/experiment.py hockey_ai_analytics/v12/development.json
/private/tmp/hockey-analytics-env/bin/python hockey_ai_analytics/v12/experiment.py hockey_ai_analytics/v12/validation.json
/private/tmp/hockey-analytics-env/bin/python hockey_ai_analytics/v12/build_report.py
```

The experiment reuses saved image-only embeddings if the source-image, track, encoder provenance and evidence-code signature matches. References are not part of that cache. Raw source data, V10/V11 helper code, inherited configs and model weights are required; this directory is not a self-contained product release.

For intentional full detector reproduction, `prepare_validation.py` downloads the declared public interval and `detect_validation.py` performs the fresh tracker run. The latter refuses to overwrite `validation/tracks.csv`. The encoder setup script downloads official upstream source and weights; preserve the existing pinned provenance rather than rerunning it against a changing upstream branch.

`review.html` links the three-way videos. Green boxes are predictions; magenta boxes are publisher references used only in rendering/evaluation. Each variant has coordinate/state CSV, full evidence JSON and evaluation CSV.

Keep the consumed validation interval as future diagnostic material, not an unseen test for a later tuned candidate. Overnight automation stays paused; no original `sources/` files or V10/V11 result archives are modified.
