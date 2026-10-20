# Research history and decisions

All historical reports are observations from their original run, not current readiness claims. File references in them may point to excluded local artifacts. Do not regenerate consumed test sets and call them unseen.

| Version | Work | Outcome / reading |
|---|---|---|
| V4–V5 | Color/motion tracking and neural person tracking | Initial hypotheses, identity not established; root and V5 QA |
| V6 | Manual selection, conservative gate and local app | Active workflow; uncertainty requires explicit reconfirmation |
| V7 | Hockey-specific objects and public clips | Detection does not establish persistent identity; V7 QA |
| V8 | Higher detail, slicing, tracker/segmenter consensus | Additional layers did not establish robust identity; tools review |
| V9 | Multiple matches, motion and jersey-number evidence | OCR/appearance evidence remains uncertain; V9 QA |
| V10 | Overnight tracker, SAM2/fusion, appearance and manual-policy comparisons | Improvements failed to generalize; no tracker promoted; morning report |
| V11 | Causal identity recovery | Development gains, holdout false reconnections; not deployed |
| V12 | Team evidence plus DINOv2 embeddings | Errors reduced mainly by abstention; coverage collapsed; not deployed |
| V13 | Hockey metric head trained on small multi-view set | Failed held-out retrieval; not deployed |
| V14 | Expanded multi-view training and review interface | Trained model worse than generic multi-view baseline; not deployed |
| V15 | Fixed crop color audit | More correct labels accompanied more wrong labels; candidate rejected |

## Important findings

- Returning model IDs can represent a different player after a gap.
- A learned generic image descriptor is not a validated hockey personnel recognizer.
- Independent team evidence is useful, but jersey numbers/stripes and mixed-person detections confuse color extraction.
- Publisher personnel labels contain some inconsistencies; sampled assistant review is not independent human confirmation.
- Multiple views and hard negatives did not automatically improve the trained metric head.
- Abstaining more can reduce errors while making the system unusable; report both.

## Reproduction discipline

Historical scripts may download data, overwrite their output folders or update a lock. Read their source and protocol before execution. Use a new experiment directory for new training/tuning. Preserve old reports and frozen hashes. `v12/setup_encoder.py` historically resolves upstream main; reproducing the recorded run requires the commit in `v12/encoder_source.json`, not today's main.

The supplied source inventory is a handoff index, not proof that missing datasets, checkpoints or generated artifacts have been restored. Original full runs are retained locally by the owner.
