"""Core inference for the Forest Change QGIS plugin (DOFA + CfC fusion, CPU).

Runs a siamese DOFA + CfC logit-stack fusion ONNX model over bi-temporal
Sentinel-2 tiles using onnxruntime (CPU by default; GPU providers if available)
and GDAL for I/O.

Model inputs (batch of 32 px chips = 320 m):
  pre, post : (B, 6, 128, 128) - bands B02,B03,B04,B08,B11,B12, reflectance/10000
  seq       : (B, 4, 9)        - 9 features (6 bands + NDVI/NBR/NDMI) for the four
                                 canonical dates, mean-pooled over the chip
Output: 4 logits - no_change / logging / windfall / fire

Two input modes:
  period  - mosaics stored as <date>_<band>.tif (+ <date>_SCL.tif) in a folder
  files   - two multi-band GeoTIFFs (T1, T2) with 6 selected band numbers
            (seq then uses only these two dates, remaining slots are zeros)

Outputs in out_dir: classes.tif, changes.gpkg, summary.json
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
CHANGE_FRAC = 0.10 # chip is 'changed' if at least this fraction of pixels changed


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
        self.names = [i.name for i in self.sess.get_inputs()]

    @property
    def provider(self):
        return self.sess.get_providers()[0]

    def run(self, pre, post, seq):
        return self.sess.run(None, {"pre": pre, "post": post, "seq": seq})[0]


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


def _read_band1(path, c0, r0, w, h):
    ds = _gdal_open(path)
    arr = ds.GetRasterBand(1).ReadAsArray(c0, r0, w, h)
    del ds
    return arr


def _read_stack(path, band_idx, c0, r0, w, h):
    ds = _gdal_open(path)
    out = np.zeros((len(band_idx), h, w), dtype=np.float32)
    for i, b in enumerate(band_idx):
        out[i] = ds.GetRasterBand(b).ReadAsArray(c0, r0, w, h)
    del ds
    return out


def _read_mosaic6(mosaic_dir, date, c0, r0, w, h):
    out = np.zeros((6, h, w), dtype=np.float32)
    for i, b in enumerate(BANDS6):
        ds = _gdal_open(os.path.join(mosaic_dir, f"{date}_{b}.tif"))
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
    else:
        ref = t1_path
    gt, W, H, proj = _grid_from(ref)

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
    change_px = (np.abs(fb_map[6] - fa_map[6]) > CHANGE_THR) | (np.abs(fb_map[7] - fa_map[7]) > CHANGE_THR)
    change_px &= ~invalid

    model = Model(model_path)
    cls = np.zeros((h, w), dtype=np.uint8)
    coords = [(r, c) for r in range(0, h - CHIP + 1, CHIP) for c in range(0, w - CHIP + 1, CHIP)]
    if not coords:
        coords = [(max(0, h - CHIP), max(0, w - CHIP))]

    buf_a, buf_b, buf_s, buf_pos = [], [], [], []
    for i, (r, c) in enumerate(coords):
        chip_a = A[:, r:r + CHIP, c:c + CHIP]
        chip_b = B[:, r:r + CHIP, c:c + CHIP]
        if chip_a.shape[1] < CHIP or chip_a.shape[2] < CHIP:
            continue
        # change is decided by the algorithm (per-pixel mask), the network only
        # classifies the type among chips that contain enough changed pixels
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
        if len(buf_a) >= BATCH or i == len(coords) - 1:
            logits = model.run(np.stack(buf_a).astype(np.float32),
                               np.stack(buf_b).astype(np.float32),
                               np.stack(buf_s).astype(np.float32))
            for (rr, cc), k in zip(buf_pos, logits.argmax(1)):
                k = int(k)
                if k == 0:      # network says "no change" but the gate says changed:
                    k = 1 + int(np.argmax(logits[0, 1:]))   # take the best type
                cls[rr:rr + CHIP, cc:cc + CHIP] = k
            buf_a, buf_b, buf_s, buf_pos = [], [], [], []
        if progress and (i % 25 == 0 or i == len(coords) - 1):
            progress(int(100 * (i + 1) / len(coords)))
    cls[invalid] = 255

    cls_path = os.path.join(out_dir, "classes.tif")
    _write_classes(cls_path, cls, gt, c0, r0, w, h, proj)
    gpkg = os.path.join(out_dir, "changes.gpkg")
    _polygonize(cls_path, gpkg, proj)
    summary = {"mode": mode, "period": period, "provider": model.provider,
               "chip": CHIP, "scale_m": gt[1],
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
