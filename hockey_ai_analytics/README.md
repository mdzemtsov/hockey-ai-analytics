# Hockey AI Analytics — Pilot 001E

**FAIL / NOT VALIDATED.** Сначала прочитать QA_report.md.
Тестовый RED_TEST_01 не идентифицирован как Матвей.

Исходный MOV и V3 найдены и скопированы в evidence/. Исходного CSV V3 нет:
v3_digitized.csv восстановлен из нарисованного овала. Предварительный эталон:
30 визуальных опорных кадров и 394 интерполированные позиции с просмотром
каждого кадра; отдельный ручной клик на каждом кадре и независимое человеческое
ревью не выполнены. Происхождение строк сохранено, сомнительные кадры исключены.

## Повторить

Python 3.13.3, macOS ARM64. В отдельном окружении установить requirements.txt.
Из каталога проекта выполнить последовательно:

```sh
python make_review.py
python run_benchmark.py
python finalize_benchmark.py
python validate_outputs.py
python -m unittest discover -s tests -v
```

Команды воспроизводят уже записанные решения визуального аудита, а не заменяют
новую проверку человеком. На другой версии декодера возможны небольшие отличия.

## Артефакты

- results/qa_comparison.mp4 — предварительный эталон / V3 / V4, исходные PTS и аудио.
- results/v4_experimental.mp4 — отдельный V4 с явными отказами.
- results/reference_ground_truth_PROVISIONAL.csv — все 424 строки, provenance и неопределённость.
- manual_anchors.csv — только визуальные опорные точки.
- identity_review_intervals.csv — решения ассистента по личности и границы неопределённости.
- results/v3_digitized.csv, v4_predictions_raw.csv — предсказания до аудита.
- results/v3_audited.csv, v4_audited.csv — отдельно добавленные результаты визуального аудита.
- results/per_frame_qa.csv — ошибки и статусы для каждого кадра обеих версий.
- results/comparison.csv и comparison.json — плотные предварительные, опорные и парные метрики.
- results/run_manifest.json, output_validation.json, checksums.json — воспроизводимость и QA.
- results/v4a_predictions_raw.csv, tracker_v4a.py — первая неудачная итерация.
- evidence/review/ — покадровые листы исходника и сравнения.

evaluate.py — дополнительный общий оценщик для строго ручной bbox-разметки.
Он намеренно не принимает reviewed_interpolation как ручную разметку.
Сравнение использует finalize_benchmark.py: bbox V3 — геометрия овала,
сопоставимый IoU недоступен.

Предлагаемые релизные критерии: >=95% покрытия однозначно видимых кадров,
отсутствие подтверждённых подмен и скрытой неопределённости, независимая проверка
и отдельный тестовый клип. Это предложения, не ранее согласованные пороги.
