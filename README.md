# Forest Change Detection — QGIS plugin (CNN + length-aware CfC)

Выявление и классификация изменений лесного покрова по разновременным снимкам Sentinel-2.
**Маску изменения** считает контекстная нейросеть (CNN-энкодер + окрестность 3×3 +
двунаправленный CfC с динамическим T, val F1 0.929–0.931), **тип изменения**
(вырубка / ветровал / пожар) — TypeNet (дистиллированный DOFA-энкодер 12×768 + CfC 2×256
со сводками, macro-F1 0.870). Геообработка — GDAL/OGR, инференс — onnxruntime
(CPU или DirectML/CUDA/DirectML с авто-откатом на CPU).

Финальные модели (v0.16): `outputs`/модели — см. [forest-change-model v3.0](https://github.com/excelsllor/forest-change-model)
(там же послойные описания `MODEL_MASK_ARCH.md`/`MODEL_TYPES_ARCH.md`, блок-схемы и код).

Схемы: [маска](https://github.com/excelsllor/forest-change-model/blob/master/docs/10_mask_final_bicfcpn_big.png) ·
[TypeNet](https://github.com/excelsllor/forest-change-model/blob/master/docs/11_typenet_final.png).

## Как работает

1. **Маска (модель)**: для каждого чипа 32 px (320 м) строится ряд по датам —
   сырой объём (≤8 дат, 6 каналов, reflectance/10000, clip 0..1.5), dt (годы) и doy (sin/cos/год).
   Малый энкодер (1.41M, общие веса по датам) даёт признак 256-d на дату; из признаков соседних
   чипов собирается **окрестность 3×3**; CfC (128, двунаправленный, **length-aware**: среднее по
   валидным шагам, обратный проход с последней даты) + сводки mean/last−first → бинарно изменение/нет.
   T динамическое (2–8); в плагине для мозаик используются 2–4 даты. Нелесные пиксели
   (SCL «не растительность»/вода, NDVI<0.3 на крайних датах) → 255.
2. **Тип (модель)**: для каждого изменённого чипа TypeNet из двух нейросетей (энкодер + CfC):
   `typenet_encoder.onnx` — CnnDofaLike 7.58M: по каждой дате признак 768-d (λ-стем +
   12 depthwise-separable слоёв, дилатации [1,2,4,8]×3, RF≈185, GAP);
   `typenet_cfc.onnx` — CfC 2×256 + сводки mean/last−first → 3 класса: logging / windfall / fire.
   T динамическое (2–8), точные длины без паддинга; CfC на CPU (Loop на DirectML некорректен).
3. Неизменённые чипы пропускаются без прогона сети типов (`no_change` не классифицируется).

## Возможности

- Режимы: **period** (мозаики `<дата>_<band>.tif` + `<дата>_SCL.tif`) и **files** (мультиканальные
  GeoTIFF по датам или T1/T2, номера каналов).
- Выходы: `mask.tif` (1/0/255), `classes.tif` (0 no_change, 1 logging, 2 windfall, 3 fire),
  `changes.gpkg` (полигоны с типом и площадью), `cells.gpkg` (клетки 320 м: координаты, тип,
  уверенность, даты), `summary.json`.

## Установка

Требуется QGIS 3.16+ (проверено на 3.44 LTR, Python 3.12).

1. скачайте `forest_change_plugin_v0.16.0.zip` из
   [Releases плагина](https://github.com/excelsllor/forest-change-qgis/releases);
2. QGIS → *Модули → Управление модулями → Установить из ZIP* → выберите файл;
3. включите модуль «Forest Change Detection (DOFA+CfC)».

**Зависимости (один раз):**
```powershell
C:\Users\<user>\AppData\Local\Programs\OSGeo4W\apps\Python312\python.exe -m pip install onnxruntime
# опционально GPU: -m pip install onnxruntime-directml
```

**Модели (52 МБ) входят в ZIP:** `masknet_encoder.onnx` (5.5 МБ), `masknet_cfc.onnx` (4.9 МБ),
`typenet_encoder.onnx` (30.3 МБ), `typenet_cfc.onnx` (10.9 МБ). Чекпоинты и код —
[forest-change-model v3.0](https://github.com/excelsllor/forest-change-model).

## Как проверить

1. Откройте QGIS, загрузите снимки → включите модуль → панель **Forest Change**.
2. Режим **«Загруженные снимки (слоты)»**: выберите слои проекта (даты), при необходимости добавьте слоты.
   - Каналы подставляются автоматически (S2 L2A → 2,3,4,8,11,12; 6-канальный стек → 1..6).
3. **Провайдер**: Авто / CPU / CUDA / DirectML (ошибки — авто-откат на CPU).
4. Укажите папку вывода и файлы моделей → **«1. Посчитать маску (модель)»**, затем
   **«2. Посчитать изменение по маске (модель)»** (или всё сразу).
5. Маска рисуется красным на прозрачном; классы по типам (logging красный, windfall золотой,
   fire маджента); **Identify** по клетке — координаты, тип, уверенность.
6. Файлы: `mask.tif`, `classes.tif`, `changes.gpkg`, `cells.gpkg`, `summary.json`.

**Ориентир (p3-мозаика, 2025-07→2026-06, 98 тайлов):** маска 14 112 чипов (144.5 тыс. га)
за 59–65 с; типы 13 385 клеток — 177 с cold / 43 с warm (кэш фич в `out_dir/cache`).

## Скорость

- Маска: ~0.5 с/тайл 1024×1024; энкодер 0.036 мс/чип (DirectML, батч 512, свёрнутый λ-стем);
  CfC на CPU (динамическое T).
- Типы: энкодер ~2.4 мс/чип-дата (DirectML); CfC на CPU; повторные прогоны используют дисковый
  кэш фич (ключ = даты + энкодер + mtime данных) — ~3 мс/клетка.
- I/O: in-memory VRT 6-в-1 для мозаик, кэш GDAL-хендлов, async-prefetch.

## Точность

**Маска изменения** (бинарно, финальная модель):

| метрика | значение |
|---|---|
| Vozh val (точные длины 2–4) | **F1 0.931** (P 0.938 / R 0.924); p1 0.925 / p2 0.960 / p3 0.906 |
| clean-val (без пересечения окон кварталов) | 0.940 |
| GT p3 на сцене через плагин | P 0.966 / R 0.952 / **F1 0.959** |
| zero-vol (без снимков) | F1 0.008 — решения из изображений |
| масштабирование | Vozh-домен; zero-shot на DISFOR/California без контекста слабый |

**Тип изменения (TypeNet; val, per-class F1):**

| датасет | logging | windfall | fire | macro |
|---|---|---|---|---|
| все | **0.922** | **0.877** | **0.983** | **0.870** |
| vozh | 0.944 | 0.698 | — | 0.547 |
| big4 (DISFOR) | 0.816 | 0.830 | 0.934 | 0.860 |
| california | — | — | 0.972 | 0.972 |
| pairs (DISFOR) | 0.921 | 0.879 | 0.959 | 0.920 |
| long (DISFOR) | 0.927 | 0.889 | 1.000 | 0.939 |

GT-чипы p3 через пайплайн плагина: recall logging 88% / windfall 94%.

## Ограничения

- Маска обучена на Vozh (целевой домен плагина); для других регионов нужен дообучение.
- Vozh windfall у TypeNet слабый (0.698) — редкий класс.
- На ложных срабатываниях маски TypeNet отвечает «windfall» (`no_change` не обучен) — доля классов
  на карте зависит от порога маски.

## Структура

```
forest_change/
  __init__.py     classFactory для QGIS
  plugin.py       регистрация пункта меню и панели
  dialog.py       диалог выбора данных/параметров
  core.py         маска (encoder+ctx+CfC) и типы (encoder+CfC), GDAL-пайплайн, выгрузка результатов
  periods.py      даты канонических периодов набора
  metadata.txt    метаданные модуля
  icon.png
  model/          masknet_encoder/masknet_cfc/typenet_encoder/typenet_cfc (.onnx)
```
