# Experimental baseline — 20 October 2026

## Working workflow

Explicit player selection; UNKNOWN latched after detection loss/ambiguity until reconfirmation; validated detection caches; source/model/code fingerprints; video/CSV exports. Coaching adds interval selection, hockey-object detection candidates, jersey-color hypotheses, marked pauses, conditional passing-lane graphics and English/Russian narration.

Coaching v1.2 checks unknown-color players, goalies and referees as lane blockers, respects explicitly requested team colors and treats a third visible color as unresolved team evidence.

## Evidence

The local working archive passed 27 unit/replay tests and service runs in both coaching modes. This clean repository includes 24 model-free synthetic/core tests; the 3 private-video replay tests are excluded. Original tracking service tests rendered all 424 frames with timestamp validation. Generated English/Russian narration decoded with non-silent audio. Browser checks covered pauses, language switching, voice controls and continuation.

Real coaching samples (system outputs, NOT true-positive counts):

| Clip | Sampled images | Two-color hypothesis | Single puck candidate | Conditional arrows |
|---|---:|---:|---:|---:|
| Original private test, 3–10 s | 7 | 1 | 3 | 0 |
| Chicago–Toronto, 0–4 s | 4 | 0 | 2 | 0 |
| Carolina–Boston, 0–4 s | 4 | 0 | 0 | 0 |

No real clip justified a tactical arrow. Arrow logic was exercised on controlled fixtures. Continuous puck tracking, possession, rink calibration, zones, tactical roles, empty-ice recognition and good/bad decision grading are not validated. Selected-player coaching only anchors the explicitly chosen frame. This is not continuous personnel identity tracking.

## Rejected experiments

V13/V14 hockey-specific identity metric heads failed validation and are not enabled. In the expanded experiment, generic multi-view retrieval reached 38.9% vs 25.5% for the trained head on the same query set; conservative calibrated acceptance rejected every query. This was oracle-box identity retrieval, not end-to-end tracking.

V15 audited 51 crops from three clips using assistant visual labels (not independent human ground truth). First sampled images were development, last images validation; matches overlap between sets. Of 24 eligible validation crops, the old color classifier had 7 correct / 3 wrong / 14 UNKNOWN. Broader torso sampling produced 12 correct / 5 wrong / 7 UNKNOWN and classified all 3 mixed-person crops. It was rejected because wrong assignments increased. No thresholds were retuned after validation.

## Limits

Workflow tests prove implementation behavior, not recognition quality. More labels and fewer gaps are not necessarily improvements. Independent human verification and a successful held-out evaluation remain prerequisites for deployment claims. Test subject is not identified as Matvey.
