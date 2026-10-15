"""Core inference for the Forest Change QGIS plugin (DOFA + CfC fusion, CPU).

Change is decided by the ALGORITHM (per-pixel |dNDVI|/|dNBR| gate); the network
only classifies the TYPE of the changed 32 px (320 m) chips.

Model inputs (batch of 32 px chips):
  pre, post : (B, 6, 128, 128) - bands B02,B03,B04,B08,B11,B12, reflectance/10000
  seq       : (B, 4, 9)        - 9 features (6 bands + NDVI/NBR/NDMI) for the four
                                 canonical dates, mean-pooled over the chip
Output: 4 logits - no_change / logging / windfall / fire

Two input modes:
  period  - mosaics stored as <date>_<band>.tif (+ <date>_SCL.tif) in a folder
  files   - two multi-band GeoTIFFs (T1, T2) with 6 selected band numbers
            (seq then uses only these two dates, remaining slots are zeros)

Outputs in out_dir: classes.tif, changes.gpkg (merged regions), cells.gpkg
(one polygon per changed 320 m cell with coordinates, type and confidence),
summary.json
"""

import json
import os

import numpy as np
from osgeo import gdal, ogr, osr

gdal.UseExceptions()

CLASSES = {0: "no_change", 1: "logging", 2: "windfall", 3: "fire"}
BANDS6 = ("B02", "B03", "B04", "B08", "B11", "B12")
SEQ_DATES = ("2023-09", "2024-05", "2025-07", "2026-06")
SCL_INVALID = [0, 1, 3, 8, 9, 10, 11]
PROVIDER_ORDER = ["DmlExecutionProvider", "CUDAExecutionProvider", "ROCMExecutionProvider",
                  "CoreMLExecutionProvider", "CPUExecutionProvider"]
CHIP = 32          # 320 m at 10 m resolution (same scale as the training chips)
SIZE = 128         # model input
BATCH = 64
SCALE = 10000.0
CHANGE_THR = 0.15  # |dNDVI| / |dNBR| threshold of the algorithmic change gate
CHANGE_FRAC = 0.02 # chip is changed if at least this fraction of pixels changed (matches the training label threshold)
MIN_AREA_HA = 0.0  # drop merged polygons smaller than this (0 = keep all)


def pick_providers():
    import onnxruntime as ort
    avail = ort.get_available_providers()
    return [p for p in PROVIDER_ORDER if p in avail] or ["CPUExecutionProvider"]


def _resize_bilinear(a, size):
    """a: (C,H,W) -> (C,size,size) bilinear (numpy only, works in QGIS Python)."""
    C, H, W = a.shape
    xi = np.linspace(0, W - 1, size)
    yi = np.linspace(0, H - 1, size)
    x0 = np.floor(xi).astype(int); x1 = np.minimum(x0 + 1, W - 1); wx = (xi - x0)[None, None, :]
    y0 = np.floor(yi).astype(int); y1 = np.minimum(y0 + 1, H - 1); wy = (yi - y0)[None, :, None]
    tmp = a[:, :, x0] * (1 - wx) + a[:, :, x1] * wx
    return tmp[:, y0, :] * (1 - wy) + tmp[:, y1, :] * wy


def _features9(bands):
    """bands: (6,h,w) reflectance (0..10000) -> (9,h,w) features."""
    b = bands.astype(np.float32) / SCALE
    b02, b03, b04, b08, b11, b12 = b
    ndvi = (b08 - b04) / np.maximum(b08 + b04, 1e-6)
    nbr = (b08 - b12) / np.maximum(b08 + b12, 1e-6)
    ndmi = (b08 - b11) / np.maximum(b08 + b11, 1e-6)
    return np.stack([b02, b03, b04, b08, b11, b12, ndvi, nbr, ndmi])


def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class Model:
    def __init__(self, path):
        import onnxruntime as ort
        self._ort = ort
        with open(path, "rb") as fh:
            self._buf = fh.read()
        providers = pick_providers()
        try:
            self.sess = ort.InferenceSession(self._buf, providers=providers)
        except Exception:
            self.sess = ort.InferenceSession(self._buf, providers=["CPUExecutionProvider"])
        self._cpu_only = False

    @property
    def provider(self):
        return self.sess.get_providers()[0]

    def _infer(self):
        return self.sess.run(None, {"pre": self._pre, "post": self._post, "seq": self._seq})[0]

    def run(self, pre, post, seq):
        self._pre, self._post, self._seq = pre, post, seq
        try:
            return self._infer()
        except Exception as exc:  # noqa: BLE001  (e.g. DirectML fails on dynamic Reshape)
            if self._cpu_only:
                raise
            print(f"[forest_change] provider {self.provider} failed ({type(exc).__name__}), "
                  "falling back to CPUExecutionProvider", flush=True)
            self.sess = self._ort.InferenceSession(self._buf, providers=["CPUExecutionProvider"])
            self._cpu_only = True
            return self._infer()


# ---------------------------------------------------------------- IO helpers
def _gdal_open(path):
    ds = gdal.Open(path)
    if ds is None:
        raise IOError(f"Cannot open {path}")
    return ds


def _grid_from(path):
    ds = _gdal_open(path)
    gt = ds.GetGeoTransform()
    return gt, ds.RasterXSize, ds.RasterYSize, ds.GetProjection()


def _assert_same_grid(paths, ref_path):
    """All inputs must share the reference geotransform and CRS (pixel-wise compare)."""
    gt0, _, _, proj0 = _grid_from(ref_path)
    for p in paths:
        if p is None or not os.path.exists(p):
            continue
        gt, _, _, proj = _grid_from(p)
        same = (abs(gt[0] - gt0[0]) < 1e-6 and abs(gt[1] - gt0[1]) < 1e-6
                and abs(gt[3] - gt0[3]) < 1e-6 and abs(gt[5] - gt0[5]) < 1e-6
                and proj.strip() == proj0.strip())
        if not same:
            raise ValueError(
                "Rasters are not on the same grid (geotransform/CRS differ): "
                f"{os.path.basename(p)}. Reproject/resample all inputs to one grid first.")


def _read_band1(path, c0, r0, w, h):
    ds = _gdal_open(path)
    arr = ds.GetRasterBand(1).ReadAsArray(c0, r0, w, h)
    del ds
    return arr


def _read_stack(path, band_idx, c0, r0, w, h):
    ds = _gdal_open(path)
    n = ds.RasterCount
    bad = [b for b in band_idx if b < 1 or b > n]
    if bad:
        del ds
        raise ValueError(
            f"{os.path.basename(path)}: запрошены каналы {list(band_idx)}, "
            f"но в файле только {n} канал(ов).\n"
            "Укажите корректные номера: для 6-канального стека — 1,2,3,4,5,6; "
            "для Sentinel-2 L2A — 2,3,4,8,11,12 (B02,B03,B04,B08,B11,B12).")
    out = np.zeros((len(band_idx), h, w), dtype=np.float32)
    for i, b in enumerate(band_idx):
        out[i] = ds.GetRasterBand(b).ReadAsArray(c0, r0, w, h)
    del ds
    return out


def guess_band_idx(path):
    """Sensible default band indices for a multi-band raster.

    >= 12 bands -> Sentinel-2 L2A order (B02,B03,B04,B08,B11,B12 = 2,3,4,8,11,12);
    6..11 bands -> the first six bands; otherwise clamp to the available count.
    """
    ds = _gdal_open(path)
    n = ds.RasterCount
    del ds
    if n >= 12:
        return (2, 3, 4, 8, 11, 12)
    if n >= 6:
        return (1, 2, 3, 4, 5, 6)
    return tuple(min(i, n) for i in (1, 2, 3, 4, 5, 6))


def band_count(path):
    ds = _gdal_open(path)
    n = ds.RasterCount
    del ds
    return n


def _read_mosaic6(mosaic_dir, date, c0, r0, w, h):
    out = np.zeros((6, h, w), dtype=np.float32)
    for i, b in enumerate(BANDS6):
        p = os.path.join(mosaic_dir, f"{date}_{b}.tif")
        if not os.path.exists(p):
            raise ValueError(
                f"В папке мозаик нет файла «{os.path.basename(p)}».\n"
                f"Папка: {mosaic_dir}\nОжидаются файлы <дата>_<канал>.tif для каналов "
                f"{', '.join(BANDS6)} и <дата>_SCL.tif по датам {', '.join(SEQ_DATES)}.")
        ds = _gdal_open(p)
        out[i] = ds.GetRasterBand(1).ReadAsArray(c0, r0, w, h)
        del ds
    return out


# ---------------------------------------------------------------- main engine
def predict(mode, model_path, out_dir, period=None, mosaic_dir=None, t1_path=None,
            t2_path=None, band_idx=None, aoi_bbox=None, progress=None):
    """mode: 'period' or 'files'. Returns summary dict."""
    os.makedirs(out_dir, exist_ok=True)
    if mode == "period":
        try:
            from . import periods
        except Exception:
            import periods
        date1, date2 = periods.PERIOD_DATES[period]
        ref = os.path.join(mosaic_dir, f"{date1}_B08.tif")
        check = [os.path.join(mosaic_dir, f"{d}_{b}.tif") for d in SEQ_DATES for b in BANDS6]
    else:
        date1, date2 = "T1", "T2"
        ref = t1_path
        check = [t1_path, t2_path]
    gt, W, H, proj = _grid_from(ref)

    # all inputs must be pixel-aligned with the reference (same CRS and geotransform)
    _assert_same_grid(check, ref)

    if aoi_bbox:
        xmin, ymin, xmax, ymax = aoi_bbox
        c0 = max(0, int((xmin - gt[0]) / gt[1]))
        r0 = max(0, int((ymax - gt[3]) / gt[5]))
        c1 = min(W, int((xmax - gt[0]) / gt[1]))
        r1 = min(H, int((ymin - gt[3]) / gt[5]))
    else:
        c0, r0, c1, r1 = 0, 0, W, H
    w, h = c1 - c0, r1 - r0

    if progress:
        progress(1)
    if mode == "period":
        A = _read_mosaic6(mosaic_dir, date1, c0, r0, w, h)
        B = _read_mosaic6(mosaic_dir, date2, c0, r0, w, h)
        feats = []
        for date in SEQ_DATES:
            try:
                feats.append(_features9(_read_mosaic6(mosaic_dir, date, c0, r0, w, h)))
            except Exception:
                feats.append(np.zeros((9, h, w), np.float32))
        seq_map = np.stack(feats)                        # (4,9,h,w)
    else:
        idx = band_idx or (1, 2, 3, 4, 5, 6)
        A = _read_stack(t1_path, idx, c0, r0, w, h)
        B = _read_stack(t2_path, idx, c0, r0, w, h)
        f1 = _features9(A)
        f2 = _features9(B)
        zero = np.zeros_like(f1)
        seq_map = np.stack([f1, f2, zero, zero])         # 2 dates only

    invalid = np.zeros((h, w), dtype=bool)
    invalid |= (A[3] == 0) | (B[3] == 0)                 # B08 nodata
    if mode == "period":
        for date in (date1, date2):
            p = os.path.join(mosaic_dir, f"{date}_SCL.tif")
            if os.path.exists(p):
                invalid |= np.isin(_read_band1(p, c0, r0, w, h), SCL_INVALID)

    # algorithmic change mask (per pixel): |dNDVI| or |dNBR| above the threshold
    fa_map = _features9(A)
    fb_map = _features9(B)
    change_px = ((np.abs(fb_map[6] - fa_map[6]) > CHANGE_THR) |
                 (np.abs(fb_map[7] - fa_map[7]) > CHANGE_THR)) & ~invalid
    if os.environ.get("FC_DEBUG"):
        print(f"[core] window c0={c0} r0={r0} {w}x{h}; invalid={int(invalid.sum())}; "
              f"change_px={int(change_px.sum())}; dNDVI mean={float((fb_map[6]-fa_map[6]).mean()):+.3f}", flush=True)

    model = Model(model_path)
    cls = np.zeros((h, w), dtype=np.uint8)
    coords = [(r, c) for r in range(0, h - CHIP + 1, CHIP) for c in range(0, w - CHIP + 1, CHIP)]
    if not coords:
        coords = [(max(0, h - CHIP), max(0, w - CHIP))]
    cells = []                                            # (row, col, cls, conf)

    buf_a, buf_b, buf_s, buf_pos = [], [], [], []

    def flush():
        """Run the network on the pending chips and record classes/cells."""
        if not buf_a:
            return
        logits = model.run(np.stack(buf_a).astype(np.float32),
                           np.stack(buf_b).astype(np.float32),
                           np.stack(buf_s).astype(np.float32))
        probs = _softmax(np.asarray(logits, dtype=np.float32))
        for idx_cell, (rr, cc) in enumerate(buf_pos):
            k = int(logits[idx_cell].argmax())
            if k == 0:      # network says "no change" but the gate says changed:
                k = 1 + int(np.argmax(logits[idx_cell, 1:]))   # take the best type
            conf = float(probs[idx_cell, k])
            cls[rr:rr + CHIP, cc:cc + CHIP] = k
            cells.append((rr, cc, k, conf))
        buf_a.clear(); buf_b.clear(); buf_s.clear(); buf_pos.clear()

    for i, (r, c) in enumerate(coords):
        chip_a = A[:, r:r + CHIP, c:c + CHIP]
        chip_b = B[:, r:r + CHIP, c:c + CHIP]
        if chip_a.shape[1] < CHIP or chip_a.shape[2] < CHIP:
            continue
        frac = float(change_px[r:r + CHIP, c:c + CHIP].mean())
        if frac < CHANGE_FRAC:
            cls[r:r + CHIP, c:c + CHIP] = 0
            if progress and (i % 25 == 0 or i == len(coords) - 1):
                progress(int(100 * (i + 1) / len(coords)))
            continue
        buf_a.append(_resize_bilinear(chip_a, SIZE) / SCALE)
        buf_b.append(_resize_bilinear(chip_b, SIZE) / SCALE)
        buf_s.append(seq_map[:, :, r:r + CHIP, c:c + CHIP].mean(axis=(2, 3)))   # (4,9)
        buf_pos.append((r, c))
        if len(buf_a) >= BATCH:
            flush()
        if progress and (i % 25 == 0 or i == len(coords) - 1):
            progress(int(100 * (i + 1) / len(coords)))
    flush()                                                   # do not lose the tail
    cls[invalid] = 255

    cls_path = os.path.join(out_dir, "classes.tif")
    _write_classes(cls_path, cls, gt, c0, r0, w, h, proj)
    gpkg = os.path.join(out_dir, "changes.gpkg")
    _polygonize(cls_path, gpkg, proj)
    cells_path = os.path.join(out_dir, "cells.gpkg")
    _write_cells(cells_path, cells, gt, c0, r0, proj, period, date1, date2)
    summary = {"mode": mode, "period": period, "provider": model.provider,
               "chip": CHIP, "scale_m": gt[1], "change_thr": CHANGE_THR,
               "cells": len(cells),
               "areas_ha": {CLASSES[k]: float((cls == k).sum() * gt[1] * abs(gt[5]) / 1e4) for k in (1, 2, 3)},
               "classes_tif": cls_path, "changes_gpkg": gpkg, "cells_gpkg": cells_path}
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


def _write_cells(path, cells, gt, c0, r0, proj, period, date1, date2):
    """One polygon per changed 320 m cell with coordinates, type and confidence."""
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    if os.path.exists(path):
        os.remove(path)
    dst = ogr.GetDriverByName("GPKG").CreateDataSource(path)
    layer = dst.CreateLayer("cells", srs, ogr.wkbPolygon)
    for name, typ in (("cell_id", ogr.OFTInteger), ("row", ogr.OFTInteger), ("col", ogr.OFTInteger),
                      ("x_center", ogr.OFTReal), ("y_center", ogr.OFTReal),
                      ("cls", ogr.OFTInteger), ("type", ogr.OFTString), ("conf", ogr.OFTReal),
                      ("period", ogr.OFTString), ("date1", ogr.OFTString), ("date2", ogr.OFTString)):
        layer.CreateField(ogr.FieldDefn(name, typ))
    defn = layer.GetLayerDefn()
    for cid, (r, c, k, conf) in enumerate(cells):
        x0 = gt[0] + (c0 + c) * gt[1]
        y0 = gt[3] + (r0 + r) * gt[5]
        x1 = x0 + CHIP * gt[1]
        y1 = y0 + CHIP * gt[5]
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for xx, yy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)):
            ring.AddPoint(xx, yy)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        feat = ogr.Feature(defn)
        feat.SetGeometry(poly)
        feat.SetField("cell_id", cid)
        feat.SetField("row", int(r))
        feat.SetField("col", int(c))
        feat.SetField("x_center", round((x0 + x1) / 2.0, 2))
        feat.SetField("y_center", round((y0 + y1) / 2.0, 2))
        feat.SetField("cls", int(k))
        feat.SetField("type", CLASSES.get(int(k), str(k)))
        feat.SetField("conf", round(conf, 3))
        if period is not None:
            feat.SetField("period", str(period))
        feat.SetField("date1", str(date1))
        feat.SetField("date2", str(date2))
        layer.CreateFeature(feat)
        feat = None
    dst = None


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
    dst = ogr.GetDriverByName("GPKG").CreateDataSource(gpkg_path)
    layer = dst.CreateLayer("changes", srs, ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("cls", ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn("type", ogr.OFTString))
    layer.CreateField(ogr.FieldDefn("area_ha", ogr.OFTReal))
    gdal.Polygonize(band, mem.GetRasterBand(1), layer, 0, [], callback=None)
    for feat in layer:
        c = feat.GetField("cls")
        area = round(feat.GetGeometryRef().GetArea() / 1e4, 3)
        if MIN_AREA_HA > 0 and area < MIN_AREA_HA:
            layer.DeleteFeature(feat.GetFID())
            continue
        feat.SetField("type", CLASSES.get(c, str(c)))
        feat.SetField("area_ha", area)
        layer.SetFeature(feat)
    dst = None

