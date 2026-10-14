from qgis.core import QgsApplication


def classFactory(iface):
    from .plugin import ForestChangePlugin
    return ForestChangePlugin(iface)
