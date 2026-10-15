import os
import traceback

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox, QPushButton, QLabel,
    QLineEdit, QProgressBar, QFileDialog, QSpinBox, QHBoxLayout, QWidget,
    QApplication, QMessageBox,
)
from qgis.PyQt.QtGui import QColor
from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer, QgsMapLayer

from . import periods

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
DEFAULT_MODEL = os.path.join(MODEL_DIR, "forest_change_dofa128.onnx")
BANDS6 = ("B02", "B03", "B04", "B08", "B11", "B12")


class ForestChangeDialog(QDialog):
    def __init__(self, iface):
        super().__init__(iface.mainWindow())
        self.iface = iface
        self.setWindowTitle("Forest Change Detection (DOFA)")
        self.resize(560, 460)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        form = QFormLayout()

        self.mode = QComboBox()
        self.mode.addItems(["Готовые периоды (мозаики)", "Свои снимки (T1/T2 GeoTIFF)"])
        self.mode.currentIndexChanged.connect(self._update_mode)
        form.addRow("Режим", self.mode)

        self.mosaic_dir = self._path_row()
        form.addRow("Папка мозаик", self.mosaic_dir[0])
        self.period = QComboBox()
        self.period.addItems(list(periods.PERIOD_DATES.keys()))
        form.addRow("Период", self.period)

        self.t1 = self._path_row()
        form.addRow("T1 (архив)", self.t1[0])
        self.t2 = self._path_row()
        form.addRow("T2 (текущий)", self.t2[0])
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
        form.addRow("Папка вывода", self.out[0])
        self.model = self._path_row(file_filter="ONNX (*.onnx)")
        self.model[1].setText(DEFAULT_MODEL)
        form.addRow("ONNX-модель", self.model[0])

        root.addLayout(form)
        self.status = QLabel("Готово")
        root.addWidget(self.status)
        self.bar = QProgressBar()
        root.addWidget(self.bar)

        run = QPushButton("Выявить изменения")
        run.clicked.connect(self._run)
        root.addWidget(run)
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

    def _update_mode(self):
        period_mode = self.mode.currentIndex() == 0
        for w in (self.mosaic_dir[0], self.period):
            w.setEnabled(period_mode)
        for w in (self.t1[0], self.t2[0], self.bands.parentWidget()):
            w.setEnabled(not period_mode)

    def _aoi_bbox(self):
        lid = self.aoi.currentData()
        if not lid:
            return None
        layer = QgsProject.instance().mapLayer(lid)
        ext = layer.extent()
        return (ext.xMinimum(), ext.yMinimum(), ext.xMaximum(), ext.yMaximum())

    def _run(self):
        model = self.model[1].text().strip()
        out_dir = self.out[1].text().strip()
        if not os.path.exists(model):
            QMessageBox.warning(self, "Ошибка", f"Модель не найдена:\n{model}")
            return
        if not out_dir:
            QMessageBox.warning(self, "Ошибка", "Укажите папку вывода")
            return
        try:
            from . import core
            self.status.setText("Инференс…")
            self.bar.setValue(0)
            QApplication.setOverrideCursor(1)

            def progress(p):
                self.bar.setValue(p)
                QApplication.processEvents()

            if self.mode.currentIndex() == 0:
                summary = core.predict(
                    "period", model, out_dir, period=self.period.currentText(),
                    mosaic_dir=self.mosaic_dir[1].text().strip(),
                    aoi_bbox=self._aoi_bbox(), progress=progress)
            else:
                summary = core.predict(
                    "files", model, out_dir, t1_path=self.t1[1].text().strip(),
                    t2_path=self.t2[1].text().strip(),
                    band_idx=tuple(sp.value() for sp in self.b_spins),
                    aoi_bbox=self._aoi_bbox(), progress=progress)
            self._load_results(summary)
            self.status.setText(
                f"Готово. Клеток: {summary.get('cells', 0)}. Площади (га): " +
                ", ".join(f"{k}={v:.1f}" for k, v in summary["areas_ha"].items()))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка", f"{exc}\n\n{traceback.format_exc()}")
            self.status.setText("Ошибка: " + str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _load_results(self, summary):
        project = QgsProject.instance()
        rl = QgsRasterLayer(summary["classes_tif"], "Forest changes (classes)")
        if rl.isValid():
            project.addMapLayer(rl)
        vl = QgsVectorLayer(summary["changes_gpkg"], "Forest changes (polygons)", "ogr")
        if vl.isValid():
            project.addMapLayer(vl)
            self._style_vector(vl)
        cells = summary.get("cells_gpkg")
        if cells and os.path.exists(cells):
            cl = QgsVectorLayer(cells, "Forest changes (cells 320 m)", "ogr")
            if cl.isValid():
                project.addMapLayer(cl)
                self._style_vector(cl)
        self.iface.mapCanvas().refresh()

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
