# Third-party sources and asset boundaries

This repository contains first-party project code, configuration and reports. It does not redistribute downloaded vendor repositories, model weights or datasets. Project scripts can reference or acquire these external dependencies:

- Ultralytics YOLO / BoT-SORT integration: https://github.com/ultralytics/ultralytics
- HockeyAI model: https://huggingface.co/SimulaMet-HOST/HockeyAI
- HockeyAI dataset: https://huggingface.co/datasets/SimulaMet-HOST/HockeyAI
- VIP-HTD public hockey benchmark: https://github.com/harshap-ai/VIP-HTD
- DINOv2: https://github.com/facebookresearch/dinov2 (recorded commit in V12 encoder_source.json)
- SAM2: https://github.com/facebookresearch/sam2
- Jersey number/OCR dependencies: see V9 source code and vendor manifest references in the historical report.
- General support-lane coaching reference: https://www.usahockey.com/news_article/show/775908

Review upstream licenses and dataset permissions before downloading, training, commercial use or redistribution. Availability on the internet is not a blanket redistribution license. This document does not claim a license audit or grant rights to third-party assets. No open-source license for the first-party project is granted by this public handoff.
