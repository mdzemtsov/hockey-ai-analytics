# 🚀 20 — Pilot Runbook: first video, QA, outputs and stop conditions

[Notion source](https://app.notion.com/p/3fec8554a5d681fb9dbefcda133b8eea?pvs=204) · [Index](../README.md)

> Historical Notion snapshot. Design targets and older status claims are not current implementation guarantees. See [current baseline](../../BASELINE_QA.md). Child-identifying names, team name and individual profile details have been generalized for public sharing.

Status: ready as a written test plan; no ingestion or video processing has been performed. Do not claim operational automation.

## Pilot goal

Produce a reliable, manually reviewed three-episode prototype from 5–10 minutes of authorized game footage, then scale to ten episodes and a full game only if recognition and tactical quality pass.

## Before video upload

1. Confirm filming/export rights and that sharing to any processing provider is permitted.
2. Select stable footage; prefer visible jersey and rink context.
3. Record team color, the selected player's jersey number, one clear reference frame, game format and approximate shifts if known.
4. Choose restricted input/output Drive folders and retention policy.
5. Verify available compute runtime, video format and size constraints before promising automated processing.

## Stage A — media ingest

A1 source file ID/checksum; A2 duration/FPS/resolution; A3 audio availability; A4 camera cuts and possible embedded replays; A5 rink line visibility; A6 metadata/privacy scan. Reject corrupt/unlicensed material.

## Stage B — identity and event labeling

B1 identify candidate player tracks; B2 verify with jersey and parent reference; B3 record identity switches/occlusions; B4 sample puck visibility; B5 propose events with source start/end; B6 mark insufficient evidence rather than guessing.

## Stage C — hockey review

C1 choose three high-evidence episodes; C2 independently separate observation, decision, execution, result; C3 compare alternative decisions against visible options; C4 check U9 age applicability and current rules; C5 human youth-coach review and parent identity signoff.

## Stage D — media production

D1 source clip with lead-in; D2 freeze frame; D3 player circle/arrow clearly labeled as illustrative if not calibrated; D4 Russian narration script; D5 subtitles; D6 render; D7 manual QA for wrong-player annotations, false puck claims, alignment and audio; D8 save private outputs.

## Stage E — training loop

E1 choose max two themes; E2 match to reviewed exercise IDs; E3 generate 5–10 min optional session and coach note; E4 add passport entry with uncertainty; E5 define what comparable observation would be sought next game.

## Required artifacts

pilot\_manifest.json, identity\_review.csv, event\_candidates.csv, coaching\_[review.md](http://review.md), annotated\_3\_[clips.mp](http://clips.mp)4, parent\_[summary.md](http://summary.md), optional\_exercise\_[plan.md](http://plan.md), qa\_[log.md](http://log.md). File names are specification, not existing files.

## Stop conditions

- Player identity ambiguous in selected event.
- Key puck/teammate claim unsupported by visible frames.
- Rights/consent unresolved.
- Render fails or mislabels person.
- Age-unsafe recommendation or diagnosis.
- Child-facing negative conclusion lacks human review.

## Proposed pass/fail

Pass: 3 clips with verified identity and source timestamps, no unsupported claims, playable annotated video, approved age-appropriate feedback and a valid linked exercise. Fail: any stop condition or missing mandatory artifact. Exact numeric tracking accuracy thresholds to be established with labeled baseline, not invented.

## Expansion

After successful 3-clip pilot: ten-episode compilation → full-game index → three-game benchmark → stable chat-triggered workflow. Each expansion is separately tested and documented.
