import os

from qgis.PyQt.QtGui import QIcon
try:  # Qt6 (QGIS 4): QAction lives in QtGui
    from qgis.PyQt.QtGui import QAction
except ImportError:  # Qt5 (QGIS 3)
    from qgis.PyQt.QtWidgets import QAction


class ForestChangePlugin:
    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.dialog = None

    def initGui(self):
        self.action = QAction("Forest Change", self.iface.mainWindow())
        self.action.setToolTip("Detect and classify forest changes (logging / windfall / fire)")
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&Forest Change", self.action)

    def unload(self):
        if self.action is not None:
            self.iface.removePluginMenu("&Forest Change", self.action)
            self.iface.removeToolBarIcon(self.action)
            self.action = None

    def run(self):
        from .dialog import ForestChangeDialog
        if self.dialog is None:
            self.dialog = ForestChangeDialog(self.iface)
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
