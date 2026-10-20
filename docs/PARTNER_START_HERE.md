# Partner handoff — start here

## What this project is

An experimental local hockey-video review tool. Two intended products share the same evidence pipeline: (1) follow a manually selected player and discuss decisions; (2) review a team's play. The current system provides a working review interaction, not reliable autonomous hockey intelligence.

## Consolidated knowledge base

The [Notion archive](notion/README.md) includes all 40 linked Hockey AI Analytics pages: coaching frameworks, bibliographies, exercise libraries, training logic and engineering reports. It is a public-safe snapshot with a coverage manifest and attachment exceptions.

## Five-minute orientation

1. Read [baseline QA](BASELINE_QA.md): what was tested and what remains unproven.
2. Read [architecture](ARCHITECTURE.md) and [research history](RESEARCH_HISTORY.md).
3. Follow [setup](SETUP.md) to run the app with separately obtained weights and your own footage.
4. Use [the demo checklist](DEMO_CHECKLIST.md) to assess the workflow and record failures.
5. Pick an item from [GitHub issues](https://github.com/mdzemtsov/hockey-ai-analytics/issues). Treat research claims as hypotheses until independently validated.

## Working today

- Local upload, manual player box, cautious tracking and manual reconfirmation after uncertainty.
- Team/individual coaching review modes for intervals up to 30 seconds.
- Player, referee, goalie, goal and puck detection candidates; tentative jersey colors.
- Pauses with graphic overlays, English/Russian transcripts and synthesized narration.
- Conditional support-lane rules, currently too little reliable evidence to produce arrows in tested real clips.
- JSON/CSV export, source/model/code fingerprints, core test suite and GitHub CI.

## Not solved

Reliable same-player identity through occlusion; stable team assignment; continuous puck tracking and possession; rink geometry and empty ice; player roles; tactically correct positive/negative judgments. Do not sell or demonstrate these as established capabilities.

## What is in the repository

The active V6 app, first-party earlier trackers, public-data acquisition scripts, V13/V14 training and evaluation code, the rejected V15 color experiment, configurations, protocols, aggregate results, QA reports, historical dependency snapshots, and synthetic tests. See [source inventory](SOURCE_INVENTORY.json).

## What stays outside Git

Original/private videos, player crops, raw per-frame annotations and detections, feature arrays, model weights, generated review media, downloaded third-party repositories and credentials. These are intentionally absent; old reports may refer to them. A source script being present does not mean every historical experiment can run on a fresh clone without its declared external inputs.

The repository is public and can be read without an invitation. Write access still requires an explicit collaborator invitation. Private video files are not published.

## Suggested division of work

- Product/coach reviewer: independently label a small set of clear and ambiguous episodes; define what would count as a sound coaching explanation.
- CV engineer: measure team, puck and identity errors on those frozen examples; handle UNKNOWN explicitly.
- App engineer: improve persisted jobs, asset/setup checks and review usability without altering validation labels.

The first useful deliverable is a reproducible, independently reviewed benchmark—not a more confident voice-over.
