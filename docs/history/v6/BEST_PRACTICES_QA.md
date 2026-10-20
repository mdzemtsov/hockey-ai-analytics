# Best practices integration — 20 October 2026

Status: implemented and workflow-tested; identity quality NOT qualified.

## Changes

- Explicit player selection remains required. After a missed detection, overlap, appearance inconsistency or motion uncertainty, output stays UNKNOWN until explicit reconfirmation. Returning model IDs cannot recover automatically.
- Manual confirmation must be a boolean, and ambiguous selections cannot override the safety gate. Earlier gaps remain empty. Later anchors never influence earlier inference.
- Review buttons open at uncertainty onset (or the recorded confirmation for a confirmed interval), instead of the end of a potentially video-long gap.
- Coverage is explicitly distinct from accuracy. Rendered boxes say PLAYER HYPOTHESIS. Team, puck and tactical analysis remain unavailable; experimental identity heads are disabled.
- Detection cache reuse requires matching source/model/configuration/code/dependency fingerprints, frame count and payload hash. Old unversioned caches are invalidated. Original V5 demo detections are retained as a hashed legacy input, not represented as newly inferred data.
- Exported provenance includes policy, hashes and runtime versions. Original sources and frozen research files unchanged.

## Verification

16 unit/replay tests passed, including persistent UNKNOWN after loss/overlap, explicit rebind, no retrospective gap filling, ambiguous confirmation rejection, true boolean requirement and stale/corrupt cache rejection. JavaScript syntax check passed.

Local service test on port 8767 passed its 9 workflow checks: authorization/origin protection, invalid selection, frame access, complete analysis, manual confirmation, media range delivery and render validation. Test job `7d18432a9e5342cdba3b2f3852ec83fb`. All 424 frames rendered; maximum PTS error 0.00013333333333420683 seconds. Marker coverage 23.95% after first selection. This is workflow coverage, NOT recognition accuracy. Identity accuracy remains null.

Results: `work/jobs/7d18432a9e5342cdba3b2f3852ec83fb/` contains result.mp4, target.csv, selection.json, report.json, render_validation.json and provenance.json. Repeat with tests/service_smoke.py against a newly started server; this creates a separate job. Full visual browser interaction was not rerun in this update.

## Limits

No claim that false reconnections are eliminated: an undetected switch inside an uninterrupted segment is still possible. Latched UNKNOWN trades continuity for explicit human control; previous V10 experiments showed this can severely reduce coverage. Reconfirmation represents the user's statement, not independent ground truth. Strong hockey identity recognition still needs independently confirmed labels and a successful held-out test. V13/V14 failed and remain excluded. The test player is not identified as Matvey. Automation remains paused.
