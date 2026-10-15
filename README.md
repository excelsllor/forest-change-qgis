# Forest Change Detection — QGIS plugin (DOFA)

Автоматическое выявление и классификация изменений лесного покрова
(no_change / logging / windfall / fire) по разновременным снимкам Sentinel-2.
Внутри — **siamese DOFA** (ViT-B/16, torchgeo), обученный на чипах 320 м,
работает на **CPU** через `onnxruntime` (GPU-провайдеры DirectML/CUDA/ROCm
подхватываются автоматически, если доступны). Геообработка — GDAL/OGR.

## Возможности

- Два режима входа:
  - **period** — мозаики Sentinel-2 в папке (`<дата>_<band>.tif` + `<дата>_SCL.tif`);
  - **files** — два multi-band GeoTIFF (T1, T2) с 6 выбранными каналами.
- Инференс чипами 32 px (320 м): 6 каналов **B02,B03,B04,B08,B11,B12** →
  bilinear 32→128 px → siamese DOFA → 4 класса.
- Маскирование nodata и облаков по SCL (класс 255).
- Выходные данные:
  - `classes.tif` — растр классов (0 no_change, 1 logging, 2 windfall, 3 fire, 255 nodata);
  - `changes.gpkg` — полигоны изменений с атрибутами `type`, `area_ha`;
  - `summary.json` — площади по классам (га) и использованный провайдер.

## Установка

Требуется QGIS 3.16+ (проверено на 3.44 LTR, Python 3.12).

**Вариант A — из ZIP:** QGIS → *Модули → Управление модулями → Установить из ZIP*.

**Вариант B — вручную:** скопировать папку `forest_change` в
`%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\` и включить модуль.

Зависимость (CPU-инференс):

```powershell
<OSGeo4W>\apps\Python312\python.exe -m pip install onnxruntime
```

## Модель

Модель **не входит в репозиторий** (426 МБ, лимит GitHub — 100 МБ на файл).
Скачайте `forest_change_dofa128.onnx` из репозитория с моделями:
[`forest-change-model` → Releases](https://github.com/excelsllor/forest-change-model/releases)
и положите в `forest_change/model/` (рядом — `forest_change_dofa128.json`).

Параметры входа: 6 каналов **B02,B03,B04,B08,B11,B12**, reflectance/10000,
чип 32 px → 128 px, siamese pre/post, выход — 4 логита.

## Точность

**Чипы (held-out кварталы, сбалансированная выборка, 12 088 чипов):**
acc **0.886**, macro-F1 **0.856**
(no_change 0.886, logging 0.922, windfall 0.893, fire 0.722).

**Полная карта (40 held-out кварталов периода 2025-07→2026-06, 1.86 млн пикселей):**
acc **0.839**, macro-F1 **0.380**
(no_change 0.91, logging 0.441, windfall 0.057, fire 0.112).

Ограничение: модель — классификатор чипов 320 м, а разметка попиксельная;
крупные вырубки определяются уверенно, разреженный ветровал и мелкие очаги
пожара на карте теряются (для них нужна попиксельная сегментация).

## Ограничения

- Ветровал на карте слаб (разреженная структура, чип 320 м «размывает» его).
- Пожар: мало размеченных пикселей; независимого теста нет.
- CPU-инференс: ~1–3 мин на тайл 1280×1280 (чипами 32 px, батчами по 64).

## Структура

```
forest_change/
  __init__.py     classFactory для QGIS
  plugin.py       регистрация пункта меню и панели
  dialog.py       диалог выбора данных/параметров
  core.py         инференс (onnxruntime + GDAL/OGR), чипы 32 px, экспорт
  periods.py      даты канонических периодов набора
  metadata.txt    метаданные модуля
  icon.png
```
