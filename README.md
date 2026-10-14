# Forest Change Detection — QGIS plugin

Автоматическое выявление и классификация изменений лесного покрова
(вырубка / ветровал / пожар) по разновременным снимкам Sentinel-2.
Внутри — нейросеть ChangeMamba (Mamba), запускаемая на CPU через `onnxruntime`
(без GPU; GPU-провайдеры DirectML / CUDA / ROCm подхватываются автоматически,
если доступны). Геообработка — GDAL/OGR.

## Возможности

- Два режима входа:
  - **period** — мозаики Sentinel-2 в папке (`<дата>_<band>.tif` + `<дата>_SCL.tif`);
  - **files** — два multi-band GeoTIFF (T1, T2) с выбранными номерами каналов.
- Тайловый инференс (tile 256, stride 128) с накоплением вероятностей.
- Маскирование nodata и облаков по SCL (класс 255).
- Выходные данные:
  - `classes.tif` — растр классов (0 no_change, 1 logging, 2 windfall, 3 fire, 255 nodata);
  - `changes.gpkg` — полигоны изменений с атрибутами `type`, `area_ha`;
  - `summary.json` — площади по классам (га) и использованный провайдер;
  - `preview.png` — превью результата.

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

Модель **не входит в репозиторий** (263 МБ, лимит GitHub — 100 МБ на файл).
Скачайте её из соседнего репозитория с моделями:
[`forest-change-model` → Releases](https://github.com/excelsllor/forest-change-model/releases/tag/v1.0)
→ `forest_change_mt_256.onnx`,
и положите в `forest_change/model/forest_change_mt_256.onnx`
(рядом — `forest_change_mt_256.json` с параметрами нормализации).

Параметры входа: 3 канала **B11, B08, B04**, reflectance/10000 → `clip(0.35)` × 255,
нормализация ImageNet (mean/std в JSON). Выход: `concat(change[2], type[3])`.

## Использование

1. Модули → **Forest Change** → панель плагина.
2. Выбрать режим (период/файлы), указать данные и AOI (опционально).
3. Указать выходную папку и ONNX-модель.
4. Нажать **Запустить**.

## Ограничения

- Ветровал слаб по precision на полной карте (спектрально близок к неизменённому лесу).
- Пожар: мало размеченных пикселей в обучающем наборе, независимого теста нет.
- CPU-инференс: ~2 мин на тайл 1280×1280 (на GPU-провайдере быстрее).

## Структура

```
forest_change/
  __init__.py     classFactory для QGIS
  plugin.py       регистрация пункта меню и панели
  dialog.py       диалог выбора данных/параметров
  core.py         инференс (onnxruntime + GDAL/OGR), тайлинг, экспорт
  periods.py      даты канонических периодов набора
  metadata.txt    метаданные модуля
  icon.png
```
