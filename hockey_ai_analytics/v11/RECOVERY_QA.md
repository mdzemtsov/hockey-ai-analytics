# Recovery v1: implemented, tested, not approved for deployment

20 October 2026. Selected-player research mode. V6 user application and V10 archive unchanged.

## Implementation

The new causal layer uses the selected first-frame box as its immutable appearance anchor. Every observation, including an unchanged tracker ID, must pass appearance, class, size and motion gates. Common motion across at least four tracked objects approximates camera translation. Overlap and competing candidates produce UNKNOWN. Reconnection requires three consecutive compatible observations of one candidate. No missing frames are filled. After 1.5 seconds without an accepted observation, explicit reselection is required.

The appearance histogram is a weak visual descriptor, not a person identity guarantee, calibrated team classifier, jersey reader or learned re-identification embedding. The common-motion estimate can be biased by player motion. Repeated wrong acceptance can reset expiry and sustain an identity error.

## Development comparison

Seven selected cases, four shared event groups, 357 case-frame observations over 199 unique original images. Publisher reference; independent human annotation verification pending.

| Case | Baseline correct/visible | Recovery correct/visible | Baseline / recovery wrong-person frames |
|---|---:|---:|---:|
| Cross blue | 14/41 | 33/41 | 23 / 0 |
| Cross white | 16/41 | 31/41 | 25 / 0 |
| Boards blue | 15/46 | 30/46 | 20 / 0 |
| Boards white | 26/46 | 38/46 | 20 / 0 |
| Edge skater | 8/32 | 29/32 | 0 / 0 |
| Edge goalie | 6/31 | 18/31 | 0 / 0 |
| Near boards | 13/41 | 7/41 | 0 / 0 |

Descriptive totals: correct visible coverage 98/278 (35.3%) → 186/278 (66.9%); wrong-person frames 88 → 0. These observations are correlated and deliberately selected from known failures; they are not a population accuracy estimate. Near-boards coverage regressed, mostly because no compatible candidate passed the gates. The goalie still has 11 misaligned output frames.

## Frozen reserved comparison

Candidate code/configuration, inherited descriptor, detector and tracker configuration were hashed before downloading CNTRL_VS_PAC_001 frames 901–1500. Two boxes were visually selected from source frame 901 before inspecting predictions or reference labels. Both methods used the same fresh tracker run beginning at frame 901; this differs from development's cached prefix tracking.

The complete 600-frame predictions for both methods were saved before evaluation read publisher GT. No parameter was changed after observing validation results. This is a 20-second temporal continuation of a previously studied match, not independent-match generalization.

| Selected target | Method | Correct / reference-visible | Wrong-person frames | Outputs while absent |
|---|---|---:|---:|---:|
| Dark left | Baseline | 74/184 (40.2%) | 94 | 0 |
| Dark left | Recovery | 80/184 (43.5%) | 48 | 0 |
| Light foreground | Baseline | 59/357 (16.5%) | 0 | 0 |
| Light foreground | Recovery | 56/357 (15.7%) | 0 | 0 |

Both initial selections bound successfully. All 600 frames were scored for both targets, including absence; absent frames are not counted as successful tracking. Human confirmations used: zero, not an estimate of how many would be needed.

**Decision: reject deployment.** The dark target retains 48 measured wrong-person frames and the light target regresses. Low false output during absence is not sufficient quality. Development gains alone would have produced a misleading conclusion.

## Failure inspection

Assistant spot inspection of source frames 960, 965, 981, 1027 and 1050 is saved in `reserved/failure_review.jpg`. This is not independent human ground truth.

At frame 1027 the green output is on a light-uniform opponent while the magenta publisher reference encloses the dark target separately. The failure is therefore not restricted to confusion between similar-uniform teammates: the histogram gate can also admit an opponent. Three consecutive compatible observations certify consistency under this weak descriptor, not identity.

The first accepted reacquisition at frame 960 is already geometrically wrong, and measured wrong-person frames start at 965. The target is later correctly reacquired at 1050, but this does not undo earlier false identity assignments. The 3.17-second first-stable-recovery statistic includes those intervening errors; it must not be presented as safe reacquisition latency.

## Evidence and reproduction

- `recovery.py`, `recovery_config.json`, `run_recovery.py`: candidate and development replay.
- `candidate_lock.json`: pre-validation frozen hashes.
- `validate_reserved.py`, `download_reserved.py`: public-source download and reserved replay.
- `recovery_summary.json`, `reserved/summary.json`: complete numerical results.
- `results/recovery/*/comparison.mp4`, `reserved/*_TEST/comparison.mp4`: baseline left, recovery right; green output, magenta publisher reference.
- Adjacent `comparison.csv` files: per-frame source number, state, candidate ID and output coordinates.
- `evaluation.csv`: per-frame scoring; wrong-person frames are not ID-switch event counts.
- `review.html`: visual evidence index.
- `recovery_video_validation.json`: full decode/frame-count checks.
- 13 unit tests pass, covering scoring, ambiguity, false same-ID continuity, class mismatch, consecutive confirmation, fragmentation recovery and expiry.

Run the commands in README, then `run_recovery.py`, `validate_reserved.py`, `render_recovery.py`, and `report.py` with the documented Python environment. Validation reuses saved tracks if present; remove only the generated reserved `tracks.csv` to deliberately repeat detector inference. The manifest records hashes, not an independent re-run of detector determinism.

## Next research change

Keep this failed candidate as a baseline. Test stronger identity evidence and explicit team classification, and inspect competing trajectories across an occlusion rather than treating a consistent local match as proof. Preserve UNKNOWN when evidence cannot separate players. Lock the next candidate before choosing another untouched validation interval. This consumed interval is now diagnostic material.

No team tactics, puck analysis, jersey-number identity or Matvey-identification claims are established by this work.
