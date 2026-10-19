import os
import re
import traceback

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox, QPushButton, QLabel,
    QLineEdit, QProgressBar, QFileDialog, QSpinBox, QHBoxLayout, QWidget,
    QApplication, QMessageBox, QCheckBox, QScrollArea,
)
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtCore import Qt, QSettings
from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer, QgsMapLayer

try:  # Qt6 (QGIS 4)
    _WAIT_CURSOR = Qt.CursorShape.WaitCursor
except AttributeError:  # Qt5
    _WAIT_CURSOR = Qt.WaitCursor

from . import periods

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
DEFAULT_MODEL = os.path.join(MODEL_DIR, "typenet_cfc.onnx")
DEFAULT_MASK_MODEL = os.path.join(MODEL_DIR, "masknet_cfc.onnx")
BANDS6 = ("B02", "B03", "B04", "B08", "B11", "B12")
SETTINGS = "forest_change"
DATE_RE = re.compile(r"(20\d{2})[-_./]?(\d{2})(?:[-_./]?(\d{2}))?")
PROVIDERS = [("Авто", "auto"), ("CPU", "cpu"), ("CUDA", "cuda"), ("DirectML", "dml")]


def parse_date(text):
    """Extract an acquisition date from a layer name / file name -> (Y, M, D) or None."""
    m = DATE_RE.search(os.path.basename(text or ""))
    if not m:
        return None
    y, mo = int(m.group(1)), int(m.group(2))
    if not 1 <= mo <= 12:
        return None
    d = int(m.group(3)) if m.group(3) and 1 <= int(m.group(3)) <= 31 else 1
    return (y, mo, d)


def date_label(dt):
    return f"{dt[0]}-{dt[1]:02d}-{dt[2]:02d}" if dt else "без даты"


class ForestChangeDialog(QDialog):
    def __init__(self, iface):
        super().__init__(iface.mainWindow())
        self.iface = iface
        self.setWindowTitle("Forest Change Detection (DOFA+CfC)")
        self.resize(680, 640)
        self.slots = []
        self._build()
        self._restore()

    # ------------------------------------------------------------- UI
    def _build(self):
        root = QVBoxLayout(self)
        form = QFormLayout()

        self.mode = QComboBox()
        self.mode.addItems(["Загруженные снимки (слоты)", "Готовые периоды (мозаики)"])
        self.mode.currentIndexChanged.connect(self._update_mode)
        form.addRow("Режим", self.mode)

        self.provider = QComboBox()
        for label, _key in PROVIDERS:
            self.provider.addItem(label)
        form.addRow("Провайдер (onnxruntime)", self.provider)

        # --- slots: loaded rasters
        self.slots_box = QVBoxLayout()
        slots_w = QWidget()
        slots_w.setLayout(self.slots_box)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(slots_w)
        scroll.setMinimumHeight(120)
        form.addRow("Снимки (выбери слои проекта)", scroll)
        row = QHBoxLayout()
        add = QPushButton("+ Добавить снимок")
        add.clicked.connect(lambda: self._add_slot())
        row.addWidget(add)
        refresh = QPushButton("Обновить список слоёв")
        refresh.clicked.connect(self._refresh_layers)
        row.addWidget(refresh)
        auto = QPushButton("Подставить два новейших")
        auto.clicked.connect(lambda: self._auto_fill(force=True))
        row.addWidget(auto)
        rw = QWidget()
        rw.setLayout(row)
        form.addRow("", rw)

        # --- mosaics mode
        self.mosaic_dir = self._path_row()
        form.addRow("Папка мозаик", self.mosaic_dir[0])
        self.period = QComboBox()
        self.period.addItems(list(periods.PERIOD_DATES.keys()))
        form.addRow("Период", self.period)

        self.bands = QHBoxLayout()
        self.b_spins = []
        defaults = (2, 3, 4, 8, 11, 12)
        for name, val in zip(BANDS6, defaults):
            sp = QSpinBox()
            sp.setRange(1, 200)
            sp.setValue(val)
            sp.setPrefix(name + ":")
            self.b_spins.append(sp)
            self.bands.addWidget(sp)
        bw = QWidget()
        bw.setLayout(self.bands)
        form.addRow("Каналы B02..B12", bw)

        self.aoi = QComboBox()
        self.aoi.addItem("Весь растр", None)
        for layer in QgsProject.instance().mapLayers().values():
            if layer.type() == QgsMapLayer.VectorLayer:
                self.aoi.addItem(layer.name(), layer.id())
        form.addRow("AOI (слой)", self.aoi)

        self.out = self._path_row(dir_mode=True)
        form.addRow("Папка вывода (маски)", self.out[0])
        self.model = self._path_row(file_filter="ONNX (*.onnx)")
        self.model[1].setText(DEFAULT_MODEL)
        form.addRow("ONNX-модель (тип)", self.model[0])
        self.mask_model = self._path_row(file_filter="ONNX (*.onnx)")
        self.mask_model[1].setText(DEFAULT_MASK_MODEL)
        form.addRow("ONNX-модель маски (3D-CNN+CfC)", self.mask_model[0])

        self.auto_apply = QCheckBox("Сразу накладывать маски на карту и приближать к ним")
        self.auto_apply.setChecked(True)
        root.addWidget(self.auto_apply)

        root.addLayout(form)
        self.status = QLabel("Готово")
        root.addWidget(self.status)
        self.bar = QProgressBar()
        root.addWidget(self.bar)

        b_mask = QPushButton("1. Посчитать маску (модель)")
        b_mask.clicked.connect(self._run_mask)
        root.addWidget(b_mask)
        b_types = QPushButton("2. Посчитать изменение по маске (модель)")
        b_types.clicked.connect(self._run_types)
        root.addWidget(b_types)
        b_all = QPushButton("Маска + изменение (всё сразу)")
        b_all.clicked.connect(self._run_all)
        root.addWidget(b_all)
        self._add_slot()
        self._add_slot()
        self._update_mode()

    def _path_row(self, dir_mode=False, file_filter="GeoTIFF (*.tif *.tiff)"):
        edit = QLineEdit()
        btn = QPushButton("…")
        row = QHBoxLayout()
        row.addWidget(edit)
        row.addWidget(btn)
        w = QWidget()
        w.setLayout(row)

        def browse():
            if dir_mode:
                p = QFileDialog.getExistingDirectory(self, "Выбрать папку")
            else:
                p, _ = QFileDialog.getOpenFileName(self, "Выбрать файл", "", file_filter)
            if p:
                edit.setText(p)
        btn.clicked.connect(browse)
        return w, edit

    # ------------------------------------------------------------ slots
    def _add_slot(self, path=None):
        combo = QComboBox()
        btn = QPushButton("−")
        row = QHBoxLayout()
        row.addWidget(combo)
        row.addWidget(btn)
        w = QWidget()
        w.setLayout(row)
        slot = {"widget": w, "combo": combo, "button": btn}
        self.slots.append(slot)

        def remove():
            if len(self.slots) <= 2:
                QMessageBox.information(self, "Слоты", "Нужно минимум 2 снимка (pre и post).")
                return
            self.slots.remove(slot)
            w.setParent(None)
            self._sync_bands()
        btn.clicked.connect(remove)
        combo.currentIndexChanged.connect(self._sync_bands)
        self.slots_box.addWidget(w)
        self._fill_combo(combo, path)
        return slot

    def _project_rasters(self):
        out = []
        for layer in QgsProject.instance().mapLayers().values():
            if layer.type() != QgsMapLayer.RasterLayer:
                continue
            path = layer.source().split("|")[0]
            dt = parse_date(layer.name()) or parse_date(path)
            out.append((dt, layer.name(), path))
        out.sort(key=lambda e: (e[0] is not None, e[0] or (0, 0, 0), e[1]), reverse=True)
        return out

    def _fill_combo(self, combo, selected=None):
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("— выбери снимок —", None)
        for dt, name, path in self._project_rasters():
            combo.addItem(f"{name}  [{date_label(dt)}]", path)
            if selected and os.path.normcase(path) == os.path.normcase(selected):
                combo.setCurrentIndex(combo.count() - 1)
        combo.blockSignals(False)

    def _refresh_layers(self):
        for slot in self.slots:
            cur = slot["combo"].currentData()
            self._fill_combo(slot["combo"], cur)
        self._auto_fill()
        self._sync_bands()

    def _auto_fill(self, force=False):
        if self.mode.currentIndex() != 0:
            return
        if not force and any(s["combo"].currentData() for s in self.slots):
            return
        dated = [e for e in self._project_rasters() if e[0]]
        if len(dated) < 2:
            return
        while len(self.slots) < 2:
            self._add_slot()
        # newest -> last slot (post), next -> first slot (pre)
        self._fill_combo(self.slots[-1]["combo"], dated[0][2])
        self._fill_combo(self.slots[0]["combo"], dated[1][2])
        self.status.setText(
            f"Подставлены: pre={dated[1][1]} [{date_label(dated[1][0])}], "
            f"post={dated[0][1]} [{date_label(dated[0][0])}]")

    def _slot_paths(self):
        paths = [s["combo"].currentData() for s in self.slots if s["combo"].currentData()]
        dated = [(parse_date(os.path.basename(p)) or (0, 0, 0), p) for p in paths]
        dated.sort(key=lambda e: e[0])          # oldest first (pre), newest last (post)
        return [p for _dt, p in dated]

    def _sync_bands(self, *args):
        if getattr(self, "_loading", False):
            return
        paths = self._slot_paths()
        path = paths[0] if paths else self.mosaic_dir[1].text().strip()
        if not path or not os.path.exists(path):
            return
        try:
            from . import core
            n = core.band_count(path)
            idx = core.guess_band_idx(path)
        except Exception:
            return
        for sp, v in zip(self.b_spins, idx):
            sp.setRange(1, max(1, n))
            sp.setValue(v)
        if paths:
            self.status.setText(f"Каналов в файле: {n}; подставлены индексы {list(idx)}")

    def _update_mode(self):
        slots_mode = self.mode.currentIndex() == 0
        for slot in self.slots:
            slot["widget"].setEnabled(slots_mode)
        for w in (self.mosaic_dir[0], self.period):
            w.setEnabled(not slots_mode)
        if slots_mode:
            self._auto_fill()

    def _aoi_bbox(self):
        lid = self.aoi.currentData()
        if not lid:
            return None
        layer = QgsProject.instance().mapLayer(lid)
        ext = layer.extent()
        return (ext.xMinimum(), ext.yMinimum(), ext.xMaximum(), ext.yMaximum())

    # ------------------------------------------------------- settings
    def _restore(self):
        self._loading = True
        s = QSettings()
        self.mode.setCurrentIndex(int(s.value(f"{SETTINGS}/mode", 0)))
        key = str(s.value(f"{SETTINGS}/provider", "auto"))
        for i, (_lbl, k) in enumerate(PROVIDERS):
            if k == key:
                self.provider.setCurrentIndex(i)
        self.mosaic_dir[1].setText(s.value(f"{SETTINGS}/mosaic_dir", ""))
        p = s.value(f"{SETTINGS}/period", "")
        if p in periods.PERIOD_DATES:
            self.period.setCurrentText(p)
        bands = s.value(f"{SETTINGS}/bands", "")
        if bands:
            try:
                for sp, v in zip(self.b_spins, [int(x) for x in str(bands).split(",")]):
                    sp.setValue(v)
            except Exception:
                pass
        self.out[1].setText(s.value(f"{SETTINGS}/out", ""))
        model = s.value(f"{SETTINGS}/model", "")
        if model:
            self.model[1].setText(model)
        mm = s.value(f"{SETTINGS}/mask_model", "")
        if mm:
            self.mask_model[1].setText(mm)
        self.auto_apply.setChecked(str(s.value(f"{SETTINGS}/auto_apply", "true")).lower() == "true")
        saved_slots = [x for x in str(s.value(f"{SETTINGS}/slots", "")).split("|") if x]
        while len(self.slots) < max(2, len(saved_slots)):
            self._add_slot()
        for slot, path in zip(self.slots, saved_slots):
            self._fill_combo(slot["combo"], path)
        self._loading = False
        self._update_mode()
        self._sync_bands()

    def _save(self):
        s = QSettings()
        s.setValue(f"{SETTINGS}/mode", self.mode.currentIndex())
        s.setValue(f"{SETTINGS}/provider", PROVIDERS[self.provider.currentIndex()][1])
        s.setValue(f"{SETTINGS}/mosaic_dir", self.mosaic_dir[1].text().strip())
        s.setValue(f"{SETTINGS}/period", self.period.currentText())
        s.setValue(f"{SETTINGS}/bands", ",".join(str(sp.value()) for sp in self.b_spins))
        s.setValue(f"{SETTINGS}/out", self.out[1].text().strip())
        s.setValue(f"{SETTINGS}/model", self.model[1].text().strip())
        s.setValue(f"{SETTINGS}/mask_model", self.mask_model[1].text().strip())
        s.setValue(f"{SETTINGS}/auto_apply", "true" if self.auto_apply.isChecked() else "false")
        s.setValue(f"{SETTINGS}/slots", "|".join(str(sl["combo"].currentData() or "") for sl in self.slots))

    # ------------------------------------------------------------- run
    def _mode(self):
        return "slots" if self.mode.currentIndex() == 0 else "period"

    def _params(self):
        if self.mode.currentIndex() == 0:
            return dict(files=self._slot_paths(),
                        band_idx=tuple(sp.value() for sp in self.b_spins),
                        aoi_bbox=self._aoi_bbox(),
                        provider=PROVIDERS[self.provider.currentIndex()][1])
        return dict(period=self.period.currentText(),
                    mosaic_dir=self.mosaic_dir[1].text().strip(),
                    aoi_bbox=self._aoi_bbox(),
                    provider=PROVIDERS[self.provider.currentIndex()][1])
    def _progress(self):
        def progress(p):
            self.bar.setValue(p)
            QApplication.processEvents()
        return progress

    def _check_inputs(self):
        if self.mode.currentIndex() == 0:
            paths = self._slot_paths()
            if len(paths) < 2:
                QMessageBox.warning(self, "Ошибка",
                                    "Выберите минимум 2 снимка в слотах (pre и post).")
                return False
            return True
        if not self.mosaic_dir[1].text().strip():
            QMessageBox.warning(self, "Ошибка", "Укажите папку мозаик")
            return False
        return True

    def _run_mask(self):
        out_dir = self.out[1].text().strip()
        if not out_dir:
            QMessageBox.warning(self, "Ошибка", "Укажите папку вывода (маски)")
            return
        if not self._check_inputs():
            return
        self._save()
        try:
            from . import core
            self.status.setText("Считаю маску (модель)…")
            self.bar.setValue(0)
            QApplication.setOverrideCursor(_WAIT_CURSOR)
            summary = core.compute_mask("files" if self._mode() == "slots" else "period",
                                        out_dir, progress=self._progress(),
                                        mask_model=self.mask_model[1].text().strip(),
                                        **self._params())
            rl = QgsRasterLayer(summary["mask_tif"], "Маска изменений")
            if rl.isValid():
                QgsProject.instance().addMapLayer(rl)
                self._style_mask(rl)
                if self.auto_apply.isChecked():
                    self.iface.mapCanvas().setExtent(rl.extent())
                    self.iface.mapCanvas().refresh()
            self.status.setText(
                f"Маска готова: изменённых пикселей {summary['changed_px']} "
                f"({summary['changed_ha']:.1f} га), порог {summary['thr']}. Файл: {summary['mask_tif']}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка", f"{exc}\n\n{traceback.format_exc()}")
            self.status.setText("Ошибка: " + str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _run_types(self):
        model = self.model[1].text().strip()
        out_dir = self.out[1].text().strip()
        if not os.path.exists(model):
            QMessageBox.warning(self, "Ошибка", f"Модель не найдена:\n{model}")
            return
        if not out_dir:
            QMessageBox.warning(self, "Ошибка", "Укажите папку вывода (маски)")
            return
        if not self._check_inputs():
            return
        self._save()
        try:
            from . import core
            self.status.setText("Классифицирую тип по маске…")
            self.bar.setValue(0)
            QApplication.setOverrideCursor(_WAIT_CURSOR)
            summary = core.predict("files" if self._mode() == "slots" else "period",
                                   model, out_dir, progress=self._progress(), **self._params())
            self._load_results(summary)
            self.status.setText(
                f"Готово (маска: {summary.get('used_mask') or 'пересчитана'}). "
                f"Клеток: {summary.get('cells', 0)}. Площади (га): " +
                ", ".join(f"{k}={v:.1f}" for k, v in summary["areas_ha"].items()))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка", f"{exc}\n\n{traceback.format_exc()}")
            self.status.setText("Ошибка: " + str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _run_all(self):
        model = self.model[1].text().strip()
        out_dir = self.out[1].text().strip()
        if not os.path.exists(model):
            QMessageBox.warning(self, "Ошибка", f"Модель не найдена:\n{model}")
            return
        if not out_dir:
            QMessageBox.warning(self, "Ошибка", "Укажите папку вывода (маски)")
            return
        if not self._check_inputs():
            return
        self._save()
        try:
            from . import core
            self.status.setText("Инференс (маска + тип)…")
            self.bar.setValue(0)
            QApplication.setOverrideCursor(_WAIT_CURSOR)
            mode = "files" if self._mode() == "slots" else "period"
            m = core.compute_mask(mode, out_dir,
                                  mask_model=self.mask_model[1].text().strip(), **self._params())
            summary = core.predict(mode, model, out_dir, progress=self._progress(), **self._params())
            self._load_results(summary)
            self.status.setText(
                f"Готово. Маска: {m['changed_ha']:.1f} га. Клеток: {summary.get('cells', 0)}. "
                "Площади (га): " + ", ".join(f"{k}={v:.1f}" for k, v in summary["areas_ha"].items()))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка", f"{exc}\n\n{traceback.format_exc()}")
            self.status.setText("Ошибка: " + str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    # --------------------------------------------------------- results
    def _load_results(self, summary):
        project = QgsProject.instance()
        rl = QgsRasterLayer(summary["classes_tif"], "Изменения: типы (классы)")
        if rl.isValid():
            project.addMapLayer(rl)
            self._style_classes(rl)
        vl = QgsVectorLayer(summary["changes_gpkg"], "Изменения: контуры", "ogr")
        if vl.isValid():
            project.addMapLayer(vl)
            self._style_vector(vl)
        cells = summary.get("cells_gpkg")
        if cells and os.path.exists(cells):
            cl = QgsVectorLayer(cells, "Изменения: клетки 320 м", "ogr")
            if cl.isValid():
                project.addMapLayer(cl)
                self._style_vector(cl)
        canvas = self.iface.mapCanvas()
        if self.auto_apply.isChecked():
            if rl.isValid():
                canvas.setExtent(rl.extent())
            elif vl.isValid():
                canvas.setExtent(vl.extent())
            canvas.refresh()

    def _style_vector(self, layer):
        from qgis.core import QgsCategorizedSymbolRenderer, QgsRendererCategory, QgsSymbol
        colors = {"logging": "#ff0000", "windfall": "#ffd700", "fire": "#ff00ff"}
        cats = []
        for name, col in colors.items():
            sym = QgsSymbol.defaultSymbol(layer.geometryType())
            sym.setColor(QColor(col))
            cats.append(QgsRendererCategory(name, sym, name))
        layer.setRenderer(QgsCategorizedSymbolRenderer("type", cats))
        layer.triggerRepaint()

    @staticmethod
    def _paletted(layer, entries):
        """entries: list of (value, QColor, label) -> paletted renderer (byte rasters)."""
        from qgis.core import QgsPalettedRasterRenderer
        classes = [QgsPalettedRasterRenderer.Class(v, c, lbl) for v, c, lbl in entries]
        layer.setRenderer(QgsPalettedRasterRenderer(layer.dataProvider(), 1, classes))
        layer.triggerRepaint()

    def _style_mask(self, layer):
        """1 = red (change), 0 and 255 = fully transparent."""
        trans = QColor(0, 0, 0, 0)
        self._paletted(layer, [(0, trans, "нет изменений"),
                               (1, QColor(255, 0, 0, 255), "изменение"),
                               (255, trans, "nodata")])
        layer.setOpacity(1.0)

    def _style_classes(self, layer):
        """Types colored, no_change/nodata fully transparent."""
        trans = QColor(0, 0, 0, 0)
        self._paletted(layer, [(0, trans, "нет изменений"),
                               (1, QColor(255, 0, 0, 255), "logging"),
                               (2, QColor(255, 215, 0, 255), "windfall"),
                               (3, QColor(255, 0, 255, 255), "fire"),
                               (255, trans, "nodata")])
        layer.setOpacity(1.0)
