# Reference and split qualifications

- The identity labels are unchanged VIP-HTD publisher personnel IDs. An assistant inspected four actual sampled crops per candidate from the expanded footage, recorded team appearance and excluded referees, goalies and visibly uncertain candidates. This is not independent human confirmation of every training crop.
- “Identity-disjoint” here means distinct publisher tracks, with different source matches and team names between fitting and the Chicago–Toronto test. Historical player transfers mean disjoint teams alone do not prove disjoint real people. No roster/name cross-match was performed; do not claim verified real-person disjointness.
- The publisher dataset's original split is not a training/evaluation guarantee for this experiment. The local training and calibration split is defined in the protocol and audit manifests.
- Training uses short snippets from three broadcasts, with only a limited number of fitted track identities from three matches. It is a small metric-learning pilot, not a production-scale hockey re-identification dataset.
- Up to twelve enrollment views are real frames chosen from the first 300 test frames using publisher identity labels. They simulate a labelled enrollment gallery; the system has not autonomously proved that these views belong to one player.
- Query extraction uses publisher bounding boxes. This intentionally isolates identity matching from detector/tracker error, and is an oracle-crop advantage. Even good retrieval would not prove successful end-to-end player tracking.
- The new query interval spans 600 source frames, but scoring samples every 15th frame (40 images). Do not report it as 600 densely evaluated tracking frames.
- Publisher IDs absent from the enrollment gallery are scored separately as unknown-person probes. They can include referees or goalies; this is not a pure unknown-skater benchmark.
- The model is a newly trained 131,456-parameter projection head over a frozen general DINOv2 encoder, not a fully fine-tuned hockey vision backbone.
