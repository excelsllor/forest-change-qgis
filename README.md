# Forest Change Detection — QGIS plugin (CNN + cross-attention)

Выявление и классификация изменений лесного покрова по разновременным снимкам Sentinel-2.
**Маску изменения** считает контекстная нейросеть (CNN-энкодер + **cross-attention 3×3** +
двунаправленный CfC с динамическим T, val F1 0.929), **тип изменения**
(вырубка / ветровал / пожар) — TypeNet (дистиллированный DOFA-энкодер 12×768 +
**cross-attention readout** по датам, macro-F1 0.911). Геообработка — GDAL/OGR, инференс — onnxruntime
(CPU или DirectML/CUDA с авто-откатом на CPU).

Финальные модели (v0.17): см. [forest-change-model v4.0](https://github.com/excelsllor/forest-change-model)
(там же послойные описания `MODEL_MASK_ARCH.md`/`MODEL_TYPES_ARCH.md`, блок-схемы и код).

Схемы: [маска](https://github.com/excelsllor/forest-change-model/blob/master/docs/10_mask_final_bicfcpn_big.png) ·
[TypeNet](https://github.com/excelsllor/forest-change-model/blob/master/docs/11_typenet_final.png).

## Как работает

1. **Маска (модель)**: для каждого чипа 32 px (320 м) строится ряд по датам —
   сырой объём (≤8 дат, 6 каналов, reflectance/10000, clip 0..1.5), dt (годы) и doy (sin/cos/год).
   Малый энкодер (1.41M, общие веса по датам) даёт признак 256-d на дату; из признаков соседних
   чипов собирается **окрестность 3×3**; **CtxAttn** (центр — query, 9 соседей — keys/values,
   8 голов, маска валидности) даёт контекст 256-d; CfC (128, двунаправленный, **length-aware**:
   среднее по валидным шагам, обратный проход с последней даты) + сводки mean/last−first →
   бинарно изменение/нет. T динамическое (2–8); в плагине для мозаик 2–4 даты. Нелесные пиксели
   (SCL «не растительность»/вода, NDVI<0.3 на крайних датах) → 255.
2. **Тип (модель)**: для каждого изменённого чипа TypeNet из двух нейросетей (энкодер + readout):
   `typenet_encoder.onnx` — CnnDofaLike 7.58M: по каждой дате признак 768-d (λ-стем +
   12 depthwise-separable слоёв, дилатации [1,2,4,8]×3, RF≈185, GAP);
   `typenet_cfc.onnx` — **AttnReadout**: 2 query-токена cross-attend к датам (8 голов, маска
   валидных дат, 2 блока) + сводки mean/last−first → 3 класса: logging / windfall / fire.
   T динамическое (2–8), точные длины без паддинга; readout на CPU.
3. Неизменённые чипы пропускаются без прогона сети типов (`no_change` не классифицируется).

## Возможности

- Режимы: **period** (мозаики `<дата>_<band>.tif` + `<дата>_SCL.tif`) и **files** (мультиканальные
  GeoTIFF по датам или T1/T2, номера каналов).
- Выходы: `mask.tif` (1/0/255), `classes.tif` (0 no_change, 1 logging, 2 windfall, 3 fire),
  `changes.gpkg` (полигоны с типом и площадью), `cells.gpkg` (клетки 320 м: координаты, тип,
  уверенность, даты), `summary.json`.

## Установка

Требуется QGIS 3.16+ (проверено на 3.44 LTR, Python 3.12).

1. скачайте `forest_change_plugin_v0.17.0.zip` из
   [Releases плагина](https://github.com/excelsllor/forest-change-qgis/releases);
2. QGIS → *Модули → Управление модулями → Установить из ZIP* → выберите файл;
3. включите модуль «Forest Change Detection (DOFA+CfC)».

**Зависимости (один раз):**
```powershell
C:\Users\<user>\AppData\Local\Programs\OSGeo4W\apps\Python312\python.exe -m pip install onnxruntime
# опционально GPU: -m pip install onnxruntime-directml
```

**Модели (48 МБ) входят в ZIP:** `masknet_encoder.onnx` (5.5 МБ), `masknet_cfc.onnx` (4.7 МБ),
`typenet_encoder.onnx` (30.3 МБ), `typenet_cfc.onnx` (9.3 МБ). Чекпоинты и код —
[forest-change-model v4.0](https://github.com/excelsllor/forest-change-model).

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

**Ориентир (p3-мозаика, 2025-07→2026-06):** маска — 42–65 с; типы — ~140 с cold
(кэш фич в `out_dir/cache`).

## Скорость

- Маска: ~0.5 с/тайл 1024×1024; энкодер 0.036 мс/чип (DirectML, батч 512);
  CtxAttn + CfC на CPU (динамическое T).
- Типы: энкодер ~2.4 мс/чип-дата (DirectML); cross-attention readout на CPU; повторные прогоны
  используют дисковый кэш фич (ключ = даты + энкодер + mtime данных).
- I/O: in-memory VRT 6-в-1 для мозаик, кэш GDAL-хендлов, async-prefetch.

## Точность

**Маска изменения** (бинарно, финальная модель):

| метрика | значение |
|---|---|
| Vozh val (точные длины 2–4) | **F1 0.929** (P 0.951 / R 0.907); p1 0.922 / p2 0.959 / p3 0.917 |
| clean-val (без пересечения окон кварталов) | 0.947 |
| GT p3 через плагин (labeled chips) | P 0.982 / R 0.967 / **F1 0.975** (p3 val-подмножество 0.917) |
| zero-vol (без снимков) | F1 0.008 — решения из изображений |
| масштабирование | Vozh-домен; zero-shot на DISFOR/California без контекста слабый |

**Тип изменения (TypeNet; val, per-class F1):**

| датасет | logging | windfall | fire | macro |
|---|---|---|---|---|
| все | **0.947** | **0.938** | **0.973** | **0.911** |
| vozh | 0.949 | 0.693 | — | 0.547 |
| big4 (DISFOR) | 0.859 | 0.910 | 0.935 | 0.901 |
| california | — | — | 0.987 | 0.987 |
| pairs (DISFOR) | 0.964 | 0.945 | 1.000 | 0.970 |
| long (DISFOR) | 0.900 | 0.857 | 1.000 | 0.919 |

**Vozh по всем классам** (val, n=886, accuracy 0.912):

| класс | n | precision | recall | F1 |
|---|---|---|---|---|
| logging | 746 | 0.934 | 0.965 | 0.949 |
| windfall | 139 | 0.765 | 0.633 | 0.693 |
| fire | 1 | 0.000 | 0.000 | 0.000 |

GT-чипы p3 через пайплайн плагина: accuracy 0.799; logging F1 0.875 (recall 0.791),
windfall F1 0.824 (recall 0.821), fire F1 0.750 (n=4).

## Ограничения

- Маска обучена на Vozh (целевой домен плагина); для других регионов нужен дообучение.
- Vozh windfall у TypeNet слабый (0.693) — редкий класс (139 val / 291 train).
- На ложных срабатываниях маски TypeNet отвечает «windfall» (`no_change` не обучен) — доля классов
  на карте зависит от порога маски.

## Структура

```
forest_change/
  __init__.py     classFactory для QGIS
  plugin.py       регистрация пункта меню и панели
  dialog.py       диалог выбора данных/параметров
  core.py         маска (encoder+CtxAttn+CfC) и типы (encoder+AttnReadout), GDAL-пайплайн
  periods.py      даты канонических периодов набора
  metadata.txt    метаданные модуля
  icon.png
  model/          masknet_encoder/masknet_cfc/typenet_encoder/typenet_cfc (.onnx)
```
