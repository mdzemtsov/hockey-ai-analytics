# Data boundaries

Commit only reviewed application code, configuration, synthetic tests and aggregate QA reports. Raw videos, player crops, frame-level annotations, trained weights, credentials, runtime caches and generated reviews remain outside Git, including in this private repository.

The original project archive is not imported. Tests with private replay fixtures are omitted; they may be run locally in the existing workspace. Research code requiring excluded datasets must fail clearly when those inputs are absent, not invent substitute results.

Use assets.json to check separately obtained model files. A known hash establishes file identity, not a license to redistribute it or proof of model quality. Keep source and license provenance for future datasets.

Never commit tokens or put them in issue bodies, reports or logs. Before each push, inspect staged files and check for credentials, private media and annotations.
