# V9 / V9.1 — многоматчевый тест нескольких систем

Статус: исследовательский стенд. Основной пользовательский интерфейс V6 не изменён. Командная и индивидуальная аналитика разделены в `analysis_contract.json`; этот этап проверяет сопровождение, не качество тактических оценок.

Зафиксированная проверка V8: CAR–BOS, CGY–DAL, CHI–TOR, первые 300 авторских JPEG каждого матча. Исправление V9.1 после этих результатов проверяется отдельно на STL–SJ, первые 300 кадров. По одному вручную выбранному на первом кадре полевому игроку. Нельзя выдавать эти короткие фрагменты за полные матчи.

У CAR–BOS, CGY–DAL, STL–SJ исходная частота 60 fps (5 секунд); у CHI–TOR — 30 fps (10 секунд). Входной технический контейнер SAM проигрывается при 30 fps, но каждый исходный кадр обработан один раз в исходном порядке, без пропуска или интерполяции. SAM использует последовательность кадров, не физическое время; пороги фиксированы в кадрах. Финальные обзорные видео используют исходные fps. Сравнение устойчивости на разных скоростях съёмки остаётся отдельной задачей.

## Запуск

Окружение Python 3.13 в `requirements_lock.txt`; нужны входные веса V7 HockeyAI и V8 SAM 2.1 Tiny. Из корня проекта:

```sh
export YOLO_CONFIG_DIR="$PWD/hockey_ai_analytics/v7/runtime"
export MPLCONFIGDIR=/private/tmp/hockey-mpl-cache
export XDG_CACHE_HOME=/private/tmp/hockey-model-cache
python hockey_ai_analytics/v9/download.py
python hockey_ai_analytics/v9/download_holdout.py
python hockey_ai_analytics/v9/run_match.py CAR_VS_BOS_001
python hockey_ai_analytics/v9/motion_check.py CAR_VS_BOS_001
python hockey_ai_analytics/v9/evaluate_revision.py CAR_VS_BOS_001
```

Повторить три последних команды для CGY_VS_DAL_003, CHI_VS_TOR_004, STL_VS_SJ_001. Сами начальные выделения сохранены в selections.json; весь дальнейший эталон читается только на этапе оценки.

```sh
python hockey_ai_analytics/v9/finalize.py
python -m unittest discover -s hockey_ai_analytics/v9/tests -v
```

## Дополнительное чтение номеров

Для первой исследовательской проверки использованы 20 вырезок из первых трёх матчей, взятых из гипотез SAM каждые 30 кадров при наличии рамки. Список зафиксирован в ocr/inputs.json. Четвёртый матч не участвовал в этом OCR-прогоне. prepare_ocr.py использует только первоначальные три матча из protocol.json и воспроизводит тот же список, включая при наличии четвёртого матча.

Система авторов Koshkina/Elder получена из https://github.com/mkoshkina/jersey-number-pipeline, commit `007d54e5530a66616ed5081ca35e0028b36aadb5`. Автоматический setup.py не запускался. Использованы только архитектуры/веса чтения и читаемости; это адаптер, не полное воспроизведение авторского конвейера с позами и re-ID. Исходная лицензия CC BY-NC 3.0 сохранена в репозитории; это исследовательское подключение.

```sh
python hockey_ai_analytics/v9/download_number_models.py
python hockey_ai_analytics/v9/load_number_checkpoint.py
python hockey_ai_analytics/v9/prepare_ocr.py
python hockey_ai_analytics/v9/specialized_numbers.py
```

Legacy checkpoint сначала разобран с ограниченным набором разрешённых глобальных объектов, затем сохранены только веса и параметры. Автоматическое присвоение личности по номеру не выполняется. Даже согласованный номер — гипотеза до отдельной проверки.

Apple Vision — отдельный локальный сравнительный OCR на macOS. В ограниченной песочнице он возвращал системную ошибку; вне неё запустился успешно. Кадры не отправлялись в облако. Компиляция: `swiftc -module-cache-path /private/tmp/hockey-swift-module-cache hockey_ai_analytics/v9/jersey_ocr.swift -o hockey_ai_analytics/v9/jersey_ocr`. Запуск: `hockey_ai_analytics/v9/jersey_ocr hockey_ai_analytics/v9/ocr/inputs.json hockey_ai_analytics/v9/ocr/results.json`.

## Артефакты

В results/<матч>: tracks.csv, team_tracks.csv, fixed_id.csv, sam.csv, hybrid.csv, revised.csv, motion_checks.csv, metrics.json, revised_metrics.json, review.mp4. Внешние исходники и хеши — source_manifest.json и holdout_source_manifest.json. Протоколы до экспериментов — protocol.json, revision_protocol.json, extra_layers_protocol.json. Результаты чтения номеров — ocr/*results.json и *summary.json. Полный статус — QA_report.md.
