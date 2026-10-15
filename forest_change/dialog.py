import os
import traceback

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox, QPushButton, QLabel,
    QLineEdit, QProgressBar, QFileDialog, QSpinBox, QHBoxLayout, QWidget,
    QApplication, QMessageBox,
)
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtCore import Qt
from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer, QgsMapLayer

try:  # Qt6 (QGIS 4)
    _WAIT_CURSOR = Qt.CursorShape.WaitCursor
except AttributeError:  # Qt5
    _WAIT_CURSOR = Qt.WaitCursor

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
        self.mode.addItems(["Р“РѕС‚РѕРІС‹Рµ РїРµСЂРёРѕРґС‹ (РјРѕР·Р°РёРєРё)", "РЎРІРѕРё СЃРЅРёРјРєРё (T1/T2 GeoTIFF)"])
        self.mode.currentIndexChanged.connect(self._update_mode)
        form.addRow("Р РµР¶РёРј", self.mode)

        self.mosaic_dir = self._path_row()
        form.addRow("РџР°РїРєР° РјРѕР·Р°РёРє", self.mosaic_dir[0])
        self.period = QComboBox()
        self.period.addItems(list(periods.PERIOD_DATES.keys()))
        form.addRow("РџРµСЂРёРѕРґ", self.period)

        self.t1 = self._path_row()
        form.addRow("T1 (Р°СЂС…РёРІ)", self.t1[0])
        self.t2 = self._path_row()
        form.addRow("T2 (С‚РµРєСѓС‰РёР№)", self.t2[0])
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
        form.addRow("РљР°РЅР°Р»С‹ B02..B12", bw)

        self.aoi = QComboBox()
        self.aoi.addItem("Р’РµСЃСЊ СЂР°СЃС‚СЂ", None)
        for layer in QgsProject.instance().mapLayers().values():
            if layer.type() == QgsMapLayer.VectorLayer:
                self.aoi.addItem(layer.name(), layer.id())
        form.addRow("AOI (СЃР»РѕР№)", self.aoi)

        self.out = self._path_row(dir_mode=True)
        form.addRow("РџР°РїРєР° РІС‹РІРѕРґР°", self.out[0])
        self.model = self._path_row(file_filter="ONNX (*.onnx)")
        self.model[1].setText(DEFAULT_MODEL)
        form.addRow("ONNX-РјРѕРґРµР»СЊ", self.model[0])

        root.addLayout(form)
        self.status = QLabel("Р“РѕС‚РѕРІРѕ")
        root.addWidget(self.status)
        self.bar = QProgressBar()
        root.addWidget(self.bar)

        run = QPushButton("Р’С‹СЏРІРёС‚СЊ РёР·РјРµРЅРµРЅРёСЏ")
        run.clicked.connect(self._run)
        root.addWidget(run)
        self._update_mode()

    def _path_row(self, dir_mode=False, file_filter="GeoTIFF (*.tif *.tiff)"):
        edit = QLineEdit()
        btn = QPushButton("вЂ¦")
        row = QHBoxLayout()
        row.addWidget(edit)
        row.addWidget(btn)
        w = QWidget()
        w.setLayout(row)

        def browse():
            if dir_mode:
                p = QFileDialog.getExistingDirectory(self, "Р’С‹Р±СЂР°С‚СЊ РїР°РїРєСѓ")
            else:
                p, _ = QFileDialog.getOpenFileName(self, "Р’С‹Р±СЂР°С‚СЊ С„Р°Р№Р»", "", file_filter)
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
            QMessageBox.warning(self, "РћС€РёР±РєР°", f"РњРѕРґРµР»СЊ РЅРµ РЅР°Р№РґРµРЅР°:\n{model}")
            return
        if not out_dir:
            QMessageBox.warning(self, "РћС€РёР±РєР°", "РЈРєР°Р¶РёС‚Рµ РїР°РїРєСѓ РІС‹РІРѕРґР°")
            return
        try:
            from . import core
            self.status.setText("РРЅС„РµСЂРµРЅСЃвЂ¦")
            self.bar.setValue(0)
            QApplication.setOverrideCursor(_WAIT_CURSOR)

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
                f"Р“РѕС‚РѕРІРѕ. РљР»РµС‚РѕРє: {summary.get('cells', 0)}. РџР»РѕС‰Р°РґРё (РіР°): " +
                ", ".join(f"{k}={v:.1f}" for k, v in summary["areas_ha"].items()))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "РћС€РёР±РєР°", f"{exc}\n\n{traceback.format_exc()}")
            self.status.setText("РћС€РёР±РєР°: " + str(exc))
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
