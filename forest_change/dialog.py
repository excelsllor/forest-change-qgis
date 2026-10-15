import os
import re
import traceback

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox, QPushButton, QLabel,
    QLineEdit, QProgressBar, QFileDialog, QSpinBox, QHBoxLayout, QWidget,
    QApplication, QMessageBox, QCheckBox,
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
DEFAULT_MODEL = os.path.join(MODEL_DIR, "forest_change_fusion128.onnx")
BANDS6 = ("B02", "B03", "B04", "B08", "B11", "B12")
SETTINGS = "forest_change"
DATE_RE = re.compile(r"(20\d{2})[-_./]?(\d{2})(?:[-_./]?(\d{2}))?")


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
        self.resize(640, 560)
        self._build()
        self._restore()

    # ------------------------------------------------------------- UI
    def _build(self):
        root = QVBoxLayout(self)
        form = QFormLayout()

        self.mode = QComboBox()
        self.mode.addItems(["Готовые периоды (мозаики)", "Свои снимки (T1/T2 GeoTIFF)"])
        self.mode.currentIndexChanged.connect(self._update_mode)
        form.addRow("Режим", self.mode)

        # --- loaded layers shortcut
        self.layers = QComboBox()
        self.layers.addItem("— подтянуть из загруженных слоёв —", None)
        form.addRow("Слои проекта", self.layers)
        refresh = QPushButton("Обновить список слоёв")
        refresh.clicked.connect(self._refresh_layers)
        form.addRow("", refresh)
        auto = QPushButton("Подставить T1/T2 (два новейших снимка)")
        auto.clicked.connect(lambda: self._auto_pick(force=True))
        form.addRow("", auto)

        self.mosaic_dir = self._path_row()
        form.addRow("Папка мозаик", self.mosaic_dir[0])
        self.period = QComboBox()
        self.period.addItems(list(periods.PERIOD_DATES.keys()))
        form.addRow("Период", self.period)

        self.t1 = self._path_row()
        form.addRow("T1 (архив)", self.t1[0])
        self.t2 = self._path_row()
        form.addRow("T2 (текущий)", self.t2[0])
        self.t1[1].textChanged.connect(self._sync_bands)
        self.t2[1].textChanged.connect(self._sync_bands)
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
        form.addRow("ONNX-модель", self.model[0])

        self.auto_apply = QCheckBox("Сразу накладывать маски на карту и приближать к ним")
        self.auto_apply.setChecked(True)
        root.addWidget(self.auto_apply)

        root.addLayout(form)
        self.status = QLabel("Готово")
        root.addWidget(self.status)
        self.bar = QProgressBar()
        root.addWidget(self.bar)

        b_mask = QPushButton("1. Посчитать маску (алгоритм NDVI/NBR)")
        b_mask.clicked.connect(self._run_mask)
        root.addWidget(b_mask)
        b_types = QPushButton("2. Посчитать изменение по маске (модель)")
        b_types.clicked.connect(self._run_types)
        root.addWidget(b_types)
        b_all = QPushButton("Маска + изменение (всё сразу)")
        b_all.clicked.connect(self._run_all)
        root.addWidget(b_all)
        self._update_mode()
        self._refresh_layers()

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

    # ------------------------------------------------------- settings
    def _restore(self):
        self._loading = True
        s = QSettings()
        self.mode.setCurrentIndex(int(s.value(f"{SETTINGS}/mode", 0)))
        self.mosaic_dir[1].setText(s.value(f"{SETTINGS}/mosaic_dir", ""))
        p = s.value(f"{SETTINGS}/period", "")
        if p in periods.PERIOD_DATES:
            self.period.setCurrentText(p)
        self.t1[1].setText(s.value(f"{SETTINGS}/t1", ""))
        self.t2[1].setText(s.value(f"{SETTINGS}/t2", ""))
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
        self.auto_apply.setChecked(str(s.value(f"{SETTINGS}/auto_apply", "true")).lower() == "true")
        # clamp the saved band indices to the real band counts of the files
        for path in (self.t1[1].text().strip(), self.t2[1].text().strip()):
            if path and os.path.exists(path):
                try:
                    from . import core
                    n = core.band_count(path)
                    for sp in self.b_spins:
                        sp.setRange(1, max(1, n))
                        if sp.value() > n:
                            sp.setValue(n)
                except Exception:
                    pass
        self._loading = False
        self._update_mode()

    def _sync_bands(self, *args):
        """Auto-pick band indices for the chosen file (S2 L2A order for >=12 bands)."""
        if getattr(self, "_loading", False):
            return
        from . import core
        path = self.t1[1].text().strip() or self.t2[1].text().strip()
        if not path or not os.path.exists(path):
            return
        try:
            n = core.band_count(path)
            idx = core.guess_band_idx(path)
        except Exception:
            return
        for sp, v in zip(self.b_spins, idx):
            sp.setRange(1, max(1, n))
            sp.setValue(v)
        self.status.setText(f"Каналов в файле: {n}; подставлены индексы {list(idx)}")

    def _save(self):
        s = QSettings()
        s.setValue(f"{SETTINGS}/mode", self.mode.currentIndex())
        s.setValue(f"{SETTINGS}/mosaic_dir", self.mosaic_dir[1].text().strip())
        s.setValue(f"{SETTINGS}/period", self.period.currentText())
        s.setValue(f"{SETTINGS}/t1", self.t1[1].text().strip())
        s.setValue(f"{SETTINGS}/t2", self.t2[1].text().strip())
        s.setValue(f"{SETTINGS}/bands", ",".join(str(sp.value()) for sp in self.b_spins))
        s.setValue(f"{SETTINGS}/out", self.out[1].text().strip())
        s.setValue(f"{SETTINGS}/model", self.model[1].text().strip())
        s.setValue(f"{SETTINGS}/auto_apply", "true" if self.auto_apply.isChecked() else "false")

    # ---------------------------------------------------------- layers
    def _refresh_layers(self):
        self.layers.clear()
        self.layers.addItem("— подтянуть из загруженных слоёв —", None)
        entries = []
        for layer in QgsProject.instance().mapLayers().values():
            if layer.type() != QgsMapLayer.RasterLayer:
                continue
            src = layer.source()
            path = src.split("|")[0]
            dt = parse_date(layer.name()) or parse_date(path)
            entries.append((dt, layer.name(), layer.id()))
        # newest first; layers without a date go last (alphabetical)
        entries.sort(key=lambda e: (e[0] is not None, e[0] or (0, 0, 0), e[1]), reverse=True)
        for dt, name, lid in entries:
            self.layers.addItem(f"{name}  [{date_label(dt)}]", lid)
        try:
            self.layers.currentIndexChanged.disconnect()
        except Exception:
            pass
        self.layers.currentIndexChanged.connect(self._pull_layer)
        self._auto_pick(entries)

    def _auto_pick(self, entries=None, force=False):
        """Put the two newest dated raster layers into T2 (newer) and T1 (older)."""
        if self.mode.currentIndex() != 1:
            return
        if entries is None:
            entries = []
            for layer in QgsProject.instance().mapLayers().values():
                if layer.type() != QgsMapLayer.RasterLayer:
                    continue
                src = layer.source()
                path = src.split("|")[0]
                dt = parse_date(layer.name()) or parse_date(path)
                if dt:
                    entries.append((dt, layer.name(), layer.id()))
            entries.sort(key=lambda e: (e[0], e[1]), reverse=True)
        dated = [e for e in entries if e[0]]
        if len(dated) < 2:
            return
        if not force and self.t1[1].text().strip() and self.t2[1].text().strip():
            return                      # user already set them (or restored from settings)
        t2_layer = QgsProject.instance().mapLayer(dated[0][2])
        t1_layer = QgsProject.instance().mapLayer(dated[1][2])
        if t2_layer is not None:
            self.t2[1].setText(t2_layer.source().split("|")[0])
        if t1_layer is not None:
            self.t1[1].setText(t1_layer.source().split("|")[0])
        self.status.setText(
            f"Подставлены новейшие слои: T1={dated[1][1]} [{date_label(dated[1][0])}], "
            f"T2={dated[0][1]} [{date_label(dated[0][0])}]")

    def _pull_layer(self):
        lid = self.layers.currentData()
        if not lid:
            return
        layer = QgsProject.instance().mapLayer(lid)
        if layer is None:
            return
        src = layer.source()
        path = src.split("|")[0]
        if not os.path.exists(path):
            QMessageBox.information(self, "Слой", "У слоя нет файлового источника — укажите путь вручную")
            return
        if self.mode.currentIndex() == 0:
            # mosaic mode: the layer's folder usually contains <date>_<band>.tif
            self.mosaic_dir[1].setText(os.path.dirname(path))
            self.status.setText(f"Папка мозаик: {os.path.dirname(path)}")
        else:
            dt = parse_date(layer.name()) or parse_date(path)
            if not self.t1[1].text().strip():
                self.t1[1].setText(path)
            elif not self.t2[1].text().strip() or self.t2[1].text().strip() == path:
                self.t2[1].setText(path)
            else:
                self.t1[1].setText(path)
            self.status.setText(f"Снимок: {os.path.basename(path)} [{date_label(dt)}]")

    def _update_mode(self):
        period_mode = self.mode.currentIndex() == 0
        for w in (self.mosaic_dir[0], self.period):
            w.setEnabled(period_mode)
        for w in (self.t1[0], self.t2[0], self.bands.parentWidget()):
            w.setEnabled(not period_mode)
        if not period_mode:
            self._auto_pick()            # fill T1/T2 from the two newest dated layers

    def _aoi_bbox(self):
        lid = self.aoi.currentData()
        if not lid:
            return None
        layer = QgsProject.instance().mapLayer(lid)
        ext = layer.extent()
        return (ext.xMinimum(), ext.yMinimum(), ext.xMaximum(), ext.yMaximum())

    # ------------------------------------------------------------- run
    def _mode(self):
        return "period" if self.mode.currentIndex() == 0 else "files"

    def _params(self):
        if self.mode.currentIndex() == 0:
            return dict(period=self.period.currentText(),
                        mosaic_dir=self.mosaic_dir[1].text().strip(),
                        aoi_bbox=self._aoi_bbox())
        return dict(t1_path=self.t1[1].text().strip(),
                    t2_path=self.t2[1].text().strip(),
                    band_idx=tuple(sp.value() for sp in self.b_spins),
                    aoi_bbox=self._aoi_bbox())

    def _progress(self):
        def progress(p):
            self.bar.setValue(p)
            QApplication.processEvents()
        return progress

    def _run_mask(self):
        out_dir = self.out[1].text().strip()
        if not out_dir:
            QMessageBox.warning(self, "Ошибка", "Укажите папку вывода (маски)")
            return
        self._save()
        try:
            from . import core
            self.status.setText("Считаю маску (алгоритм)…")
            self.bar.setValue(0)
            QApplication.setOverrideCursor(_WAIT_CURSOR)
            summary = core.compute_mask(self._mode(), out_dir, progress=self._progress(), **self._params())
            rl = QgsRasterLayer(summary["mask_tif"], "Маска изменений")
            if rl.isValid():
                QgsProject.instance().addMapLayer(rl)
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
        self._save()
        try:
            from . import core
            self.status.setText("Классифицирую тип по маске…")
            self.bar.setValue(0)
            QApplication.setOverrideCursor(_WAIT_CURSOR)
            summary = core.predict(self._mode(), model, out_dir, progress=self._progress(), **self._params())
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
        self._save()
        try:
            from . import core
            self.status.setText("Инференс (маска + тип)…")
            self.bar.setValue(0)
            QApplication.setOverrideCursor(_WAIT_CURSOR)
            m = core.compute_mask(self._mode(), out_dir, **self._params())
            summary = core.predict(self._mode(), model, out_dir, progress=self._progress(), **self._params())
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
        rl = QgsRasterLayer(summary["classes_tif"], "Изменения: классы")
        if rl.isValid():
            try:
                rl.setOpacity(0.75)
            except Exception:
                pass
            project.addMapLayer(rl)
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
            # immediately overlay and zoom to the detected masks
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
