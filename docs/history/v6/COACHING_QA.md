# Coaching review prototype — 20 October 2026

**Status: joint workflow testing available; autonomous tactical coaching is NOT qualified.**

## Implemented

- Local upload/demo, user-chosen interval up to 30 seconds, team mode and explicit single-frame player selection.
- Existing hockey-specific model detects player, goalie, referee, goal and puck candidates. This is an experimental coaching branch; it does not replace V6's identity tracker or promote V13/V14 identity heads.
- Conservative central-jersey color hypotheses: red, blue, yellow, green, white, dark, UNKNOWN. Pixel support is not a calibrated probability. Referees and goalies are excluded from jersey-team grouping.
- Original frame coordinates, approximately one sample per second. No personnel identity inferred between samples.
- Evidence-gated, conditional passing-lane rule. A unique puck candidate, two color groups, clear nearest-player geometry, adequate separation and no detected opponent on the image line are required. A selected-player arrow must involve the explicitly selected player. This rule is an illustrative possibility, not proof of possession or a safe pass.
- Automatic playback pauses, marked frames, repeat explanation, continue/stop controls, English/Russian transcripts and locally synthesized audio. Browser speech is a fallback with an explicit unavailable notice. Voice is synthesized system speech, not a custom voice clone.
- JSON/CSV observations and provenance (source/model/code hashes, interval/config). Sample images and per-moment M4A narration remain on disk. Reopen a ready review using its job URL while that local server is running.

## Tested

24 unit/replay tests passed, including all previous gate/cache tests, known-color images, unknown crop, missing/multiple pucks, blocked conditional lane, no transfer of another player's lane to the selected player, and invalid requests. These are logic tests, not real-world accuracy validation.

Service tests exercised team and player modes repeatedly with the original clip. Both English and Russian narration were generated and served. Example audio decoded to 16.22 seconds (English, mean -16.1 dB) and 20.08 seconds (Russian, mean -16.6 dB): neither was silent. See tests/coach_audio_check.json and tests/coach_service_result.json.

Two existing public input videos were uploaded to the local server and analyzed; originals remained unchanged. See tests/coach_public_result.json.

| Clip / evaluated interval | Sampled images | Two-color hypothesis | Single puck candidate | Conditional arrows |
|---|---:|---:|---:|---:|
| Original test video, 3–10 s, team mode | 7 | 1 | 3 | 0 |
| Chicago–Toronto, 0–4 s | 4 | 0 | 2 | 0 |
| Carolina–Boston, 0–4 s | 4 | 0 | 0 | 0 |

Counts are system outputs, **not true-positive counts**. No ground-truth accuracy is asserted. All real clips abstained from tactical arrows; arrow-generation logic was tested on controlled fixtures only. Do not describe the prototype as successful automated coaching.

Browser checks: demo loading, interval submission, automatic pause, marked original frame, transcript, English/Russian switching, voice playback request, stop, continue, and opening an existing review URL. Screenshot: coach_review_verified.png. The final build is running on port 8769; earlier test servers use other ports. Audio files were decoded and checked for signal; pronunciation/naturalness is not independently rated.

## Critical limitations

- Team color grouping remains unreliable under partial views, mixed jersey colors, occlusion and detector misses. A color match is not verified membership. No jersey numbers are read here.
- Puck detection is intermittent. No continuous puck trajectory, possession, stick contact or pass event is established.
- Coordinates are image pixels, not rink meters. No camera calibration, rink-zone recognition, confirmed empty ice, attack direction, forward/defender assignment or off-screen awareness.
- Selected-player coaching is anchored only at the explicitly selected frame; other sampled frames remain identity UNKNOWN. Existing full-frame tracker is separately accessible from the page.
- No validated good/bad decision grading, blame, promised movement route or game-plan interpretation. Conditional image lines can be invalidated by sticks, depth, offside or unseen players.
- Sampling can miss the entire decisive action. Long-form narration repeats a small rule set and is not an expert interpretation of the full play.
- Reloading a job URL requires the same server session; disk outputs persist but in-memory video/job registration does not survive restart.
- Test subject is not identified as Matvey. No raw footage, crops or coordinates sent to Notion. Overnight automation remains paused.

## Next accuracy work

Collect independently checked jersey/team and puck annotations on the difficult scenes, measure detection and grouping precision/coverage, add calibrated rink geometry and temporal puck/possession evidence, then validate specific hockey rules on coach-reviewed episodes. These are prerequisites for honest positive/negative judgments and recommended skating routes.

## Coaching reference

USA Hockey, [Creating offense with zone entries and puck support](https://www.usahockey.com/news_article/show/775908), supports the general emphasis on scanning and support options. It does not validate the image-space thresholds or this prototype's output.

## Reproduce

From project root:

```sh
/private/tmp/hockey-analytics-env/bin/python -m unittest discover -s hockey_ai_analytics/v6/tests -p 'test_*.py'
/private/tmp/hockey-analytics-env/bin/python hockey_ai_analytics/v6/server.py --port 8769
# Separate terminal, while server runs:
/private/tmp/hockey-analytics-env/bin/python hockey_ai_analytics/v6/tests/coach_service.py http://127.0.0.1:8769
/private/tmp/hockey-analytics-env/bin/python hockey_ai_analytics/v6/tests/coach_public.py
```

Open `/coach`, choose a video, choose 0–30 seconds, language and mode, then build the review. In player mode capture a paused frame and draw the selection first. Read the uncertainty in each pause, then use Explain aloud or Play with pauses. Results are in work/jobs/<job_id>/.

Final regression: the original selected-player service smoke test also passed all 9 checks on port 8769, including 424 rendered frames and timestamp validation. The latest coaching service rerun included an actual detector-bound player anchor and asserted that at least one pause records USER_SELECTED_FRAME_ONLY. Notion summary: https://app.notion.com/p/3ffc8554a5d68153bc42cee27d646aae .
