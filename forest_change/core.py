"""Core inference for the Forest Change QGIS plugin.

Runs a ChangeMamba ONNX model over bi-temporal Sentinel-2 tiles using
onnxruntime (auto EP: DirectML / CUDA / ROCm / CPU) and GDAL for I/O.

Two input modes:
  period  - mosaics stored as <date>_<band>.tif (+ <date>_SCL.tif) in a folder
  files   - two multi-band GeoTIFFs (T1, T2) with selected band numbers

Outputs in out_dir: classes.tif, changes.gpkg, summary.json, preview.png
"""

import json
import os

import numpy as np
from osgeo import gdal, ogr, osr

gdal.UseExceptions()

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(3, 1, 1)
CLASSES = {0: "no_change", 1: "logging", 2: "windfall", 3: "fire"}
SCL_INVALID = [0, 1, 3, 8, 9, 10, 11]
PROVIDER_ORDER = ["DmlExecutionProvider", "CUDAExecutionProvider", "ROCMExecutionProvider",
                  "CoreMLExecutionProvider", "CPUExecutionProvider"]


def pick_providers():
    import onnxruntime as ort
    avail = ort.get_available_providers()
    return [p for p in PROVIDER_ORDER if p in avail] or ["CPUExecutionProvider"]


def _norm(raw):
    x = raw.astype(np.float32) / 10000.0
    x = np.clip(x / 0.35, 0, 1) * 255.0
    return (x - MEAN) / STD


class Model:
    def __init__(self, path):
        import onnxruntime as ort
        with open(path, "rb") as fh:
            buf = fh.read()
        providers = pick_providers()
        try:
            self.sess = ort.InferenceSession(buf, providers=providers)
        except Exception:
            self.sess = ort.InferenceSession(buf, providers=["CPUExecutionProvider"])
        self.inames = [i.name for i in self.sess.get_inputs()]

    @property
    def provider(self):
        return self.sess.get_providers()[0]

    def run(self, pre, post):
        out = self.sess.run(None, {self.inames[0]: pre, self.inames[1]: post})[0]
        return out[0]  # (5, T, T)


def _softmax(x, axis=0):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------- IO helpers
def _gdal_open(path):
    ds = gdal.Open(path)
    if ds is None:
        raise IOError(f"Cannot open {path}")
    return ds


def _read_mosaic_tile(mosaic_dir, date, bands, c0, r0, t, gsize):
    out = np.zeros((3, t, t), dtype=np.float32)
    for i, b in enumerate(bands):
        p = os.path.join(mosaic_dir, f"{date}_{b}.tif")
        ds = _gdal_open(p)
        out[i] = ds.GetRasterBand(1).ReadAsArray(c0, r0, t, t)
    return out


def _read_stack_tile(path, band_idx, c0, r0, t):
    ds = _gdal_open(path)
    out = np.zeros((3, t, t), dtype=np.float32)
    for i, b in enumerate(band_idx):
        out[i] = ds.GetRasterBand(b).ReadAsArray(c0, r0, t, t)
    return out


def _grid_from(path):
    ds = _gdal_open(path)
    gt = ds.GetGeoTransform()
    return gt, ds.RasterXSize, ds.RasterYSize, ds.GetProjection()


def _read_band1(path, c0, r0, w, h):
    ds = _gdal_open(path)
    arr = ds.GetRasterBand(1).ReadAsArray(c0, r0, w, h)
    del ds
    return arr


# ---------------------------------------------------------------- main engine
def predict(mode, model_path, out_dir, period=None, mosaic_dir=None, t1_path=None,
            t2_path=None, band_idx=(1, 2, 3), aoi_bbox=None, tile=256, stride=128,
            progress=None, tile_fn=None):
    """mode: 'period' or 'files'. Returns summary dict."""
    os.makedirs(out_dir, exist_ok=True)
    date1 = date2 = None
    if mode == "period":
        try:
            from . import periods
        except Exception:  # standalone/testing import
            import periods
        date1, date2 = periods.PERIOD_DATES[period]
        # grid from any mosaic band
        ref = os.path.join(mosaic_dir, f"{date1}_B08.tif")
    else:
        ref = t1_path
    gt, W, H, proj = _grid_from(ref)

    # AOI window
    if aoi_bbox:
        xmin, ymin, xmax, ymax = aoi_bbox
        c0 = max(0, int((xmin - gt[0]) / gt[1]))
        r0 = max(0, int((ymax - gt[3]) / gt[5]))
        c1 = min(W, int((xmax - gt[0]) / gt[1]))
        r1 = min(H, int((ymin - gt[3]) / gt[5]))
    else:
        c0, r0, c1, r1 = 0, 0, W, H
    w, h = c1 - c0, r1 - r0

    model = Model(model_path)
    acc = np.zeros((4, h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    coords = [(r, c) for r in range(0, h - tile + 1, stride) for c in range(0, w - tile + 1, stride)]
    if not coords:
        coords = [(max(0, h - tile), max(0, w - tile))]

    for i, (r, c) in enumerate(coords):
        gc, gr = c0 + c, r0 + r
        if mode == "period":
            a = _norm(_read_mosaic_tile(mosaic_dir, date1, ("B11", "B08", "B04"), gc, gr, tile, gt))
            b = _norm(_read_mosaic_tile(mosaic_dir, date2, ("B11", "B08", "B04"), gc, gr, tile, gt))
        else:
            a = _norm(_read_stack_tile(t1_path, band_idx, gc, gr, tile))
            b = _norm(_read_stack_tile(t2_path, band_idx, gc, gr, tile))
        out = model.run(a[None], b[None])
        pch = _softmax(out[:2], 0)
        pty = _softmax(out[2:], 0)
        prob = np.stack([pch[0], pch[1] * pty[0], pch[1] * pty[1], pch[1] * pty[2]])
        rr, cc = min(r, h - tile), min(c, w - tile)
        acc[:, rr:rr + tile, cc:cc + tile] += prob
        cnt[rr:rr + tile, cc:cc + tile] += 1
        if progress and (i % 25 == 0 or i == len(coords) - 1):
            progress(int(100 * (i + 1) / len(coords)))
    cnt[cnt == 0] = 1
    cls = (acc / cnt).argmax(0).astype(np.uint8)

    # nodata / cloud mask
    invalid = np.zeros((h, w), dtype=bool)
    if mode == "period":
        for date in (date1, date2):
            p = os.path.join(mosaic_dir, f"{date}_SCL.tif")
            if os.path.exists(p):
                scl = _read_band1(p, c0, r0, w, h)
                invalid |= np.isin(scl, SCL_INVALID)
        for date in (date1, date2):
            p = os.path.join(mosaic_dir, f"{date}_B08.tif")
            invalid |= _read_band1(p, c0, r0, w, h) == 0
    cls[invalid] = 255

    cls_path = os.path.join(out_dir, "classes.tif")
    _write_classes(cls_path, cls, gt, c0, r0, w, h, proj)
    gpkg = os.path.join(out_dir, "changes.gpkg")
    _polygonize(cls_path, gpkg, proj)
    summary = {"mode": mode, "period": period, "provider": model.provider,
               "tile": tile,
               "areas_ha": {CLASSES[k]: float((cls == k).sum() * gt[1] * abs(gt[5]) / 1e4) for k in (1, 2, 3)},
               "classes_tif": cls_path, "changes_gpkg": gpkg}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    return summary


def _write_classes(path, cls, gt, c0, r0, w, h, proj):
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(path, w, h, 1, gdal.GDT_Byte, options=["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform((gt[0] + c0 * gt[1], gt[1], 0, gt[3] + r0 * gt[5], 0, gt[5]))
    ds.SetProjection(proj)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(255)
    band.WriteArray(cls)
    ds = None


def _polygonize(classes_path, gpkg_path, proj):
    src = gdal.Open(classes_path)
    band = src.GetRasterBand(1)
    arr = band.ReadAsArray()
    mask_arr = ((arr > 0) & (arr < 200)).astype(np.uint8)
    mem = gdal.GetDriverByName("MEM").Create("", src.RasterXSize, src.RasterYSize, 1, gdal.GDT_Byte)
    mem.GetRasterBand(1).WriteArray(mask_arr)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    if os.path.exists(gpkg_path):
        os.remove(gpkg_path)
    ogr_drv = ogr.GetDriverByName("GPKG")
    dst = ogr_drv.CreateDataSource(gpkg_path)
    layer = dst.CreateLayer("changes", srs, ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("cls", ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn("type", ogr.OFTString))
    layer.CreateField(ogr.FieldDefn("area_ha", ogr.OFTReal))
    gdal.Polygonize(band, mem.GetRasterBand(1), layer, 0, [], callback=None)
    for feat in layer:
        c = feat.GetField("cls")
        feat.SetField("type", CLASSES.get(c, str(c)))
        feat.SetField("area_ha", round(feat.GetGeometryRef().GetArea() / 1e4, 3))
        layer.SetFeature(feat)
    dst = None
