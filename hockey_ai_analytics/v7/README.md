# Hockey AI V7 — воспроизводимый эксперимент

Главный результат и ограничения: [QA_report.md](QA_report.md).

Это отдельный исследовательский стенд, а не обновление пользовательского сервера V6. Два режима используют общие обнаружения: выбранный игрок с фиксированным ID и предложения для ручного восстановления; вся сцена с гипотезами команд. Видео не подтверждают готовность тактической аналитики.

## Повторить

Python 3.13. Окружение зафиксировано в requirements_lock.txt. Из корня этого проекта:

```sh
python -m venv .venv-hockey-v7
.venv-hockey-v7/bin/pip install -r hockey_ai_analytics/v7/requirements_lock.txt
export YOLO_CONFIG_DIR="$PWD/hockey_ai_analytics/v7/runtime"
.venv-hockey-v7/bin/python hockey_ai_analytics/v7/download_public.py
.venv-hockey-v7/bin/python hockey_ai_analytics/v7/download_frames.py
.venv-hockey-v7/bin/python hockey_ai_analytics/v7/benchmark.py baseline
.venv-hockey-v7/bin/python hockey_ai_analytics/v7/benchmark.py hockey
.venv-hockey-v7/bin/python hockey_ai_analytics/v7/object_diagnostic.py
.venv-hockey-v7/bin/python hockey_ai_analytics/v7/team_analysis.py
.venv-hockey-v7/bin/python hockey_ai_analytics/v7/evaluate_recovery.py
.venv-hockey-v7/bin/python hockey_ai_analytics/v7/finalize.py
.venv-hockey-v7/bin/python -m unittest discover -s hockey_ai_analytics/v7/tests -v
```

Нужна базовая модель `hockey_ai_analytics/v5/models/yolo11s.pt` из предыдущего этапа. Если её нет, получить официальные веса Ultralytics и проверить SHA-256 по `artifact_manifest.json` исходного проекта или журналу V5. Все остальные внешние входы загружаются download_public.py и download_frames.py. Нельзя считать этот архив автономным без весов и входных данных.

Набор кадров фиксирован: авторские JPEG 1–300 PIT_VS_SJ_001. Скрипты не используют опубликованный MP4 для привязки разметки. Не менять окно и пороги после просмотра метрик без новой версии эксперимента.

## Результаты

- `results/baseline/review_h264.mp4` и `results/hockey/review_h264.mp4`: одинаковый материал и трекер, разные модели.
- `results/teams/team_review.mp4`: гипотезы команд; числа отражают только обнаруженные объекты в кадре.
- `results/*/tracks.csv`, `selected.csv`, `metrics.json`: воспроизводимая проверка обнаружений и идентичностей.
- `results/teams/reappearance_candidates.csv`: предложения, НЕ подтверждённые личности и НЕ восстановленный трек.
- `results/teams/frame_context.csv`: честные статусы неготовых компонентов, без выдуманных тактических событий.
- `results/objects`: диагностика объектов на 40 изображениях, возможное пересечение с обучением.

Лицензии и происхождение исходников сохранены в data. Публичный доступ не означает отсутствие авторских прав; архив результатов предназначен для исследовательской проверки, лицензии данных и моделей необходимо сохранять при дальнейшем использовании.
