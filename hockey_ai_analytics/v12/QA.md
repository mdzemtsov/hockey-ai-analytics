# V12 QA — stronger evidence implemented, both new variants rejected

20 October 2026. No changes deployed to the V6 user application.

## What was tested

Added an explicit two-team upper-jersey comparison and a learned DINOv2 visual descriptor. Individual-player evidence compares the selected player's first-frame crop with competing first-frame people. Both are causal gates on the existing V11 recovery layer, including observations with an unchanged tracker ID.

The three-way comparison isolates the added identity descriptor from the team check. V11 replay reproduces every previously recorded V11 metric for both development targets exactly.

Code, thresholds, pretrained weights and detector configuration were frozen before the new validation images were downloaded. No settings were adjusted after validation. All methods share detector tracks, manual seed boxes and publisher references. All predictions are saved before future GT is read for scoring.

## Development: consumed CNTRL–PAC interval, 20 seconds

| Selected target | Variant | Correct / visible frames | Wrong-person frames |
|---|---|---:|---:|
| Dark left | V11 | 80/184 | 48 |
| Dark left | Team only | 32/184 | 0 |
| Dark left | Team + identity | 12/184 | 0 |
| Light foreground | V11 | 56/357 | 0 |
| Light foreground | Team only | 45/357 | 0 |
| Light foreground | Team + identity | 22/357 | 0 |

The new gates suppress false outputs but remove substantial correct coverage. For the dark target's 80 V11-correct observations, the full gate accepts visual evidence on only 20, rejects 23 as identity-ambiguous and rejects 37 on team evidence. Confirmation and expiry reduce actual emitted coverage further. For the light target's 56 V11-correct observations, the corresponding counts are 32, 13 and 11.

These are diagnostics on already-consumed footage, not a new holdout.

## Validation: new CAR–NYR interval, 600 frames / 10 seconds

Frames 1801–2400 were declared before downloading. Two opposing-team players were visually selected from frame 1801 before inspecting predictions or GT. A fresh tracker starts at frame 1801. This is an unseen temporal continuation of another previously studied match, not independent-match generalization.

| Selected target | Variant | Correct / visible frames | Wrong-person frames |
|---|---|---:|---:|
| Blue near | V11 | 111/600 (18.5%) | 364 |
| Blue near | Team only | 110/600 (18.3%) | 361 |
| Blue near | Team + identity | 9/600 (1.5%) | 0 |
| White right | V11 | 159/162 (98.1%) | 0 |
| White right | Team only | 159/162 (98.1%) | 0 |
| White right | Team + identity | 18/162 (11.1%) | 0 |

Both initial selections bind successfully in all variants. All 600 frames are scored for each target, including absence. No variant outputs while the reference target is absent. No manual reconfirmations were performed; this is not a measured manual-intervention requirement.

**Decision: reject both new variants for release.**

- Team evidence cannot establish individual identity. On the blue target it leaves 361 wrong-person frames.
- The full identity gate avoids measured false assignments largely by abstaining: it misses 591 of 600 visible blue-target frames. This is not a working tracker.
- Adding the learned descriptor reduces the white target from 159 to 18 correct visible frames, although the original tracker handled it well.
- A general pretrained image descriptor with one immutable view is not demonstrated to be a usable hockey identity gate by this experiment. This result does not evaluate a hockey-trained re-identification model or a manually confirmed multiview gallery.

## Visual failure review

`validation/failure_review.jpg` contains source frames 1967, 2210 and 2396. In the middle sample, green team-only output encloses a blue player by the near boards, while the magenta publisher reference encloses a different blue player near the far goal. The error passes the team check because both players have the same team appearance.

This is assistant spot review, not independent human GT verification. Publisher boxes are unchanged; no interpolated or future positions are supplied to inference.

## Verification and limitations

- Ten focused evidence-gate tests pass: opponent rejection, same-team competitor rejection, identity ambiguity, missing features, ambiguous team calibration, ablation behavior, same-ID checks, and retained occlusion evidence.
- V11 baseline metrics are reproduced exactly on the development interval.
- Four three-way comparison videos are decoded fully; each has 600 frames at the native rate.
- Source URLs and hashes, model commit and weight hashes, frozen configuration, predictions, evidence reasons and evaluation CSVs are saved.
- The dataset has two intervals, four selected targets and 1,200 unique source images. It is not evidence about whole matches, youth video, all-player identity accuracy, puck tracking or tactical understanding.
- Team labels themselves are not independently scored here. The manual two-team examples add setup information that plain V11 does not use; they are disclosed.
- Wrong-person frame counts are not ID-switch event counts. A short first recovery delay can hide later identity failures.
- Model-source download uses the official [DINOv2 repository](https://github.com/facebookresearch/dinov2); this is a general feature extractor, not a validated player-identity service.

## Recommended next experiment

Move from single-view appearance matching to **hockey-specific identity evidence across multiple views**. Use an independent, reviewed identity dataset for training and evaluation, keep same-team players as hard negative examples, and test a manually confirmed multiview gallery before permitting automatic gallery updates. An unverified recovery must never become a positive identity example.

Keep team evidence as one component of matching. It cannot resolve teammate crossings. Preserve UNKNOWN and offer explicit reconfirmation when the visual evidence cannot distinguish individuals. Do not weaken these thresholds simply to restore coverage or reuse this consumed interval as an unseen test.

The current evidence supports further research, not “ready, upload your next video.”
