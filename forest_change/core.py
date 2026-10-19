"""Core inference for the Forest Change QGIS plugin (distilled CNN + CfC, CPU).

TILED pipeline: the AOI is processed in windows (TILE_PX, a multiple of CHIP), so
only a fragment of each date is held in RAM at a time (bounded ~0.3 GB for 1024 px
tiles regardless of the raster size).

  1) compute_mask()  - MaskNet-v2 (CNN + bidirectional CfC) -> mask.tif (1/0/255)
  2) predict()       - TypeNet: per-date chips -> encoder -> CfC -> logging/windfall/fire;
                       reuses mask.tif (tiles without changes are skipped without reads)

Model inputs (raw reflectance/10000, clipped to 0..1.5 as in training):
  mask: vol (B,T,6,32,32), dt (B,T) years between consecutive kept dates,
        doy (B,T,3) = (sin, cos, year-2015); T<=8, zero padded to 8
  types: chips (M,6,32,32) -> typenet_encoder.onnx -> feats (M,768);
         feats + dt + doy -> typenet_cfc.onnx -> 3 logits (no no_change class)

Input modes:
  period  - mosaics stored as <date>_<band>.tif (+ <date>_SCL.tif) in a folder
  files   - multi-band GeoTIFFs (one per date, or T1/T2) with 6 band numbers

Outputs in out_dir: mask.tif, classes.tif, changes.gpkg, cells.gpkg, summary.json
"""

import json
import os
import time

import numpy as np
from osgeo import gdal, ogr, osr

gdal.UseExceptions()

CLASSES = {0: "no_change", 1: "logging", 2: "windfall", 3: "fire"}
BANDS6 = ("B02", "B03", "B04", "B08", "B11", "B12")
SEQ_DATES = ("2023-09", "2024-05", "2025-07", "2026-06")
SCL_INVALID = [0, 1, 3, 5, 6, 8, 9, 10, 11]   # + 5 not-vegetated (urban/bare), 6 water
FOREST_NDVI_MIN = 0.3      # a pixel must be vegetated (NDVI >= min) on at least one date
                           # to be considered forest (filters urban/bare/cropland leftovers)
PROVIDER_ORDER = ["DmlExecutionProvider", "CUDAExecutionProvider", "ROCMExecutionProvider",
                  "CoreMLExecutionProvider", "CPUExecutionProvider"]
PROVIDER_MAP = {
    "auto": PROVIDER_ORDER,
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
    "directml": ["DmlExecutionProvider", "CPUExecutionProvider"],
    "rocm": ["ROCMExecutionProvider", "CPUExecutionProvider"],
}
CHIP = 32          # 320 m at 10 m resolution (same scale as the training chips)
SCALE = 10000.0
MASK_T = 8             # fixed sequence length of the exported mask ONNX (zero padding)
MASK_BATCH = 128       # chips per model call
DT_CANON = (0.0, 0.72, 1.15, 1.0)   # years between the canonical mosaic dates
DATE_RE = None         # compiled lazily in _parse_date
CHANGE_THR = 0.15  # |dNDVI| / |dNBR| threshold of the algorithmic change gate
GATE_MODE = os.environ.get("FC_GATE", "fixed").lower()   # fixed | adaptive | both
GATE_Z = float(os.environ.get("FC_GATE_Z", "4.0"))       # robust z-score threshold (adaptive)
CHANGE_FRAC = 0.02 # chip is changed if at least this fraction of pixels changed
MIN_AREA_HA = 0.0  # drop merged polygons smaller than this (0 = keep all)
TILE_PX = int(os.environ.get("FC_TILE", "1024"))   # processing window (multiple of CHIP)
SUBTILE = os.environ.get("FC_SUBTILE", "1") == "1"  # read only the mask bbox inside a tile
BLOCK = max(CHIP, int(os.environ.get("FC_BLOCK", "128")) // CHIP * CHIP)  # mask block for reads


def pick_providers(preferred="auto"):
    """Execution providers according to the user's choice (falls back to CPU)."""
    import onnxruntime as ort
    avail = ort.get_available_providers()
    want = PROVIDER_MAP.get(str(preferred or "auto").lower(), PROVIDER_ORDER)
    got = [p for p in want if p in avail]
    if not got:
        print(f"[forest_change] провайдер '{preferred}' недоступен (есть: {avail}), использую CPU",
              flush=True)
        got = ["CPUExecutionProvider"]
    return got


def _ndvi(bands):
    """NDVI from a (6,h,w) reflectance window (band order B02,B03,B04,B08,B11,B12)."""
    b04 = bands[2].astype(np.float32)
    b08 = bands[3].astype(np.float32)
    return (b08 - b04) / np.maximum(b08 + b04, 1e-6)


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


# ---------------------------------------------------------------- tiling
def _setup(mode, period=None, mosaic_dir=None, files=None, t1_path=None, t2_path=None,
           aoi_bbox=None):
    """Reference grid, AOI window, resolved pre/post/seq inputs and files to validate."""
    pre_path = post_path = None
    seq_paths = None
    if mode == "period":
        try:
            from . import periods
        except Exception:
            import periods
        date1, date2 = periods.PERIOD_DATES[period]
        ref = os.path.join(mosaic_dir, f"{date1}_B08.tif")
        check = [os.path.join(mosaic_dir, f"{d}_{b}.tif") for d in SEQ_DATES for b in BANDS6]
    else:
        fl = [p for p in (files or []) if p]
        if not fl and t1_path and t2_path:
            fl = [t1_path, t2_path]
        fl = [p for p in fl if os.path.exists(p)]
        if len(fl) < 2:
            raise ValueError("Нужно выбрать минимум 2 снимка (pre и post) из загруженных слоёв.")
        date1, date2 = "T1", "T2"
        ref = fl[0]
        check = fl
        pre_path, post_path = fl[0], fl[-1]
        seq_paths = ([None] * (4 - min(4, len(fl))) + fl[-4:])[:4]   # up to 4 dates, padded
    gt, W, H, proj = _grid_from(ref)
    _assert_same_grid(check, ref)
    if aoi_bbox:
        xmin, ymin, xmax, ymax = aoi_bbox
        c0 = max(0, int((xmin - gt[0]) / gt[1]))
        r0 = max(0, int((ymax - gt[3]) / gt[5]))
        c1 = min(W, int((xmax - gt[0]) / gt[1]))
        r1 = min(H, int((ymin - gt[3]) / gt[5]))
    else:
        c0, r0, c1, r1 = 0, 0, W, H
    return dict(gt=gt, proj=proj, c0=c0, r0=r0, w=c1 - c0, h=r1 - r0,
                date1=date1, date2=date2, pre_path=pre_path, post_path=post_path,
                seq_paths=seq_paths, period=period, mosaic_dir=mosaic_dir)


def _tiles(w, h, tile=None):
    """Windows (x, y, tw, th) inside the AOI, aligned to the CHIP grid."""
    tile = max(CHIP, (tile or TILE_PX) // CHIP * CHIP)
    out = []
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            out.append((x, y, min(tile, w - x), min(tile, h - y)))
    return out


def _gate(A, B, invalid, thr=None):
    """Algorithmic per-pixel change mask.

    fixed    : |dNDVI| or |dNBR| above CHANGE_THR (absolute threshold)
    adaptive : robust z-score of max(|dNDVI|,|dNBR|) above GATE_Z (med + z*1.4826*MAD),
               which adapts to the local seasonal/noise level of the window
    both     : union of the two
    """
    thr = CHANGE_THR if thr is None else thr
    fa = _features9(A)
    fb = _features9(B)
    g = np.maximum(np.abs(fb[6] - fa[6]), np.abs(fb[7] - fa[7]))
    ok = ~invalid if invalid is not None else np.ones(g.shape, bool)
    if GATE_MODE == "adaptive":
        v = g[ok]
        if v.size >= 100:
            med = float(np.median(v))
            mad = float(np.median(np.abs(v - med))) + 1e-6
            gate = g > med + GATE_Z * 1.4826 * mad
        else:
            gate = g > thr
    elif GATE_MODE == "both":
        v = g[ok]
        gate = g > thr
        if v.size >= 100:
            med = float(np.median(v))
            mad = float(np.median(np.abs(v - med))) + 1e-6
            gate = gate | (g > med + GATE_Z * 1.4826 * mad)
    else:
        gate = g > thr
    return gate & ok if invalid is not None else gate


# ---------------------------------------------------------------- step 1: mask
def _parse_date(text):
    """(year, month) from a file/layer name, or None."""
    global DATE_RE
    if DATE_RE is None:
        import re
        DATE_RE = re.compile(r"(20\d{2})[-_./]?(\d{2})")
    m = DATE_RE.search(os.path.basename(str(text or "")))
    if not m:
        return None
    y, mo = int(m.group(1)), int(m.group(2))
    return (y, mo) if 1 <= mo <= 12 else None


def _select_dates(n):
    """At most MASK_T dates, first and last always kept (evenly spaced)."""
    if n <= MASK_T:
        return list(range(n))
    idx = np.linspace(0, n - 1, MASK_T).round().astype(int)
    return sorted({int(i) for i in idx})


def _doy3(label, year=True):
    """(sin DOY, cos DOY, year-2015) for a date label ('YYYY-MM' or a file path).

    year=False for the canonical mosaics (the training series used year=0 there).
    """
    import datetime
    ym = _parse_date(label)
    if ym is None:
        return np.zeros(3, np.float32)
    y, m = ym
    t = datetime.date(y, m, 15).timetuple().tm_yday
    return np.array([np.sin(2 * np.pi * t / 365.0), np.cos(2 * np.pi * t / 365.0),
                     (y - 2015.0) if year else 0.0], np.float32)


def _series_for(s, mode, dates=None):
    """Selected dates of the series: (labels, dt_years, kind, paths).

    period mode -> the canonical mosaic dates up to the period end;
    files mode  -> the user's files with real acquisition dates when parseable.
    """
    if mode == "period":
        try:
            from . import periods
        except Exception:
            import periods
        date2 = periods.PERIOD_DATES[s["period"]][1]
        labels = [d for d in SEQ_DATES if d <= date2]
        dt = np.cumsum(np.concatenate([[0.0], np.array(DT_CANON[1:len(labels)])]))
        return labels, dt, "mosaic", None
    files = [p for p in (s.get("series_files") or []) if p]
    parsed = [_parse_date(p) for p in files]
    if all(parsed):
        yrs = np.array([y + (m - 1) / 12.0 for (y, m) in parsed], np.float32)
        dt = np.concatenate([[0.0], np.diff(yrs)]).astype(np.float32)
    else:
        dt = np.concatenate([[0.0], np.ones(len(files) - 1, np.float32)])
    return files, dt, "stack", files


def _series_read(mode, s, band_idx, label, c0, r0, w, h):
    """One date window (6,h,w) reflectance + SCL window (or None)."""
    if mode == "period":
        bands = _read_mosaic6(s["mosaic_dir"], label, c0, r0, w, h)
        p = os.path.join(s["mosaic_dir"], f"{label}_SCL.tif")
        scl = _read_band1(p, c0, r0, w, h) if os.path.exists(p) else None
        return bands, scl
    idx = band_idx or (1, 2, 3, 4, 5, 6)
    return _read_stack(label, idx, c0, r0, w, h), None


class MaskMiniModel:
    """Binary change mask model (CNN encoder + bidirectional CfC, ONNX, fixed T=8)."""

    def __init__(self, path, provider="cpu"):
        import onnxruntime as ort
        self._ort = ort
        with open(path, "rb") as fh:
            self._buf = fh.read()
        providers = pick_providers(provider)
        try:
            self.sess = ort.InferenceSession(self._buf, providers=providers)
        except Exception:
            self.sess = ort.InferenceSession(self._buf, providers=["CPUExecutionProvider"])
        self._cpu_only = self.sess.get_providers() == ["CPUExecutionProvider"]

    @property
    def provider(self):
        return self.sess.get_providers()[0]

    def run(self, vol, dt, doy):
        """vol (B,T,6,32,32), dt (B,T), doy (B,T,3) -> (B,2) logits; pads to T=8."""
        B, T = vol.shape[0], vol.shape[1]
        if T < MASK_T:
            pad = MASK_T - T
            vol = np.concatenate([vol, np.zeros((B, pad, 6, 32, 32), np.float32)], 1)
            dt = np.concatenate([dt, np.zeros((B, pad), np.float32)], 1)
            doy = np.concatenate([doy, np.zeros((B, pad, 3), np.float32)], 1)
        return self.sess.run(None, {"vol": vol.astype(np.float32),
                                    "dt": dt.astype(np.float32),
                                    "doy": doy.astype(np.float32)})[0]


class EncoderModel:
    """Channel-agnostic CNN encoder (12 layers x 768, ONNX): chips -> per-date features."""

    def __init__(self, path, provider="cpu"):
        import onnxruntime as ort
        with open(path, "rb") as fh:
            buf = fh.read()
        try:
            self.sess = ort.InferenceSession(buf, providers=pick_providers(provider))
        except Exception:
            self.sess = ort.InferenceSession(buf, providers=["CPUExecutionProvider"])

    @property
    def provider(self):
        return self.sess.get_providers()[0]

    def run(self, chips):
        """chips (M,6,32,32) raw reflectance -> (M,768)."""
        return self.sess.run(None, {"vol": chips.astype(np.float32)})[0]


class TypeCfCModel:
    """TypeNet CfC (2 blocks x 256, ONNX): feats + dt + doy -> (M,3) logits (fixed T=8)."""

    def __init__(self, path, provider="cpu"):
        import onnxruntime as ort
        with open(path, "rb") as fh:
            buf = fh.read()
        try:
            self.sess = ort.InferenceSession(buf, providers=pick_providers(provider))
        except Exception:
            self.sess = ort.InferenceSession(buf, providers=["CPUExecutionProvider"])

    @property
    def provider(self):
        return self.sess.get_providers()[0]

    def run(self, feats, dt, doy):
        B, T = feats.shape[0], feats.shape[1]
        if T < MASK_T:
            pad = MASK_T - T
            feats = np.concatenate([feats, np.zeros((B, pad, 768), np.float32)], 1)
            dt = np.concatenate([dt, np.zeros((B, pad), np.float32)], 1)
            doy = np.concatenate([doy, np.zeros((B, pad, 3), np.float32)], 1)
        return self.sess.run(None, {"feats": feats.astype(np.float32),
                                    "dt": dt.astype(np.float32),
                                    "doy": doy.astype(np.float32)})[0]


def compute_mask(mode, out_dir, period=None, mosaic_dir=None, files=None, t1_path=None,
                 t2_path=None, band_idx=None, aoi_bbox=None, progress=None, thr=None,
                 provider="auto", mask_model=None, dates=None):
    """Model-based change mask -> mask.tif (1 changed chip / 0 / 255).

    MaskNet-v2 (bicfcpn_big ONNX) processes the raw-reflectance time series of every
    32 px chip (up to 8 dates, zero padded) with dt (years) and doy (sin/cos/year).
    """
    if not mask_model or not os.path.exists(mask_model):
        raise ValueError(f"Модель маски не найдена: {mask_model}")
    os.makedirs(out_dir, exist_ok=True)
    s = _setup(mode, period, mosaic_dir, files, t1_path, t2_path, aoi_bbox)
    if mode != "period":
        s["series_files"] = list(files or []) or ([t1_path, t2_path] if t1_path else [])
    labels, dt_all, kind, paths = _series_for(s, mode, dates)
    sel = _select_dates(len(labels))
    labels = [labels[i] for i in sel]
    dt_sel = np.zeros(len(sel), np.float32)
    for k in range(1, len(sel)):
        dt_sel[k] = float(dt_all[sel[k - 1] + 1:sel[k] + 1].sum())
    doy_sel = np.stack([_doy3(l, year=(kind != "mosaic")) for l in labels]) \
        if labels else np.zeros((0, 3), np.float32)
    print(f"[forest_change] маска: {len(sel)} дат из {len(dt_all)} (первая/последняя сохранены), "
          f"модель {os.path.basename(mask_model)}", flush=True)

    path = os.path.join(out_dir, "mask.tif")
    ds = _create_raster(path, s["gt"], s["c0"], s["r0"], s["w"], s["h"], s["proj"])
    band = ds.GetRasterBand(1)
    tiles = _tiles(s["w"], s["h"])
    model = MaskMiniModel(mask_model, provider)
    changed_chips = 0
    invalid_px = 0
    if progress:
        progress(1)
    buf_vol, buf_dt, buf_doy, buf_pos = [], [], [], []

    def flush():
        if not buf_vol:
            return
        logits = model.run(np.stack(buf_vol), np.stack(buf_dt), np.stack(buf_doy))
        for j, (r, c) in enumerate(buf_pos):
            if int(np.argmax(logits[j])) == 1:
                cells_changed.append((r, c))
        buf_vol.clear(); buf_dt.clear(); buf_doy.clear(); buf_pos.clear()

    for ti, (x, y, tw, th) in enumerate(tiles):
        out = np.zeros((th, tw), np.uint8)
        invalid = np.zeros((th, tw), bool)
        series = []
        for li, label in enumerate(labels):
            bands, scl = _series_read(mode, s, band_idx, label, s["c0"] + x, s["r0"] + y, tw, th)
            series.append(bands)
            if scl is not None and li in (0, len(labels) - 1):
                invalid |= np.isin(scl, SCL_INVALID)
        first, last = series[0], series[-1]
        invalid |= (first[3] == 0) | (last[3] == 0)
        if FOREST_NDVI_MIN > 0:
            invalid |= (_ndvi(first) < FOREST_NDVI_MIN) & (_ndvi(last) < FOREST_NDVI_MIN)
        cells_changed = []
        n_before = len(cells_changed)
        for r in range(0, th - CHIP + 1, CHIP):
            for c in range(0, tw - CHIP + 1, CHIP):
                if invalid[r:r + CHIP, c:c + CHIP].mean() > 0.5:
                    continue
                vol = np.zeros((len(series), 6, CHIP, CHIP), np.float32)
                for li in range(len(series)):
                    b = series[li][:, r:r + CHIP, c:c + CHIP].astype(np.float32) / SCALE
                    vol[li] = np.clip(b, 0.0, 1.5)     # raw reflectance, as in training
                buf_vol.append(vol); buf_dt.append(dt_sel.copy())
                buf_doy.append(doy_sel.copy())
                buf_pos.append((r, c))
                if len(buf_vol) >= MASK_BATCH:
                    flush()
        flush()
        for (r, c) in cells_changed:
            out[r:r + CHIP, c:c + CHIP] = 1
            changed_chips += 1
        out[invalid] = 255
        band.WriteArray(out, x, y)
        invalid_px += int(invalid.sum())
        if progress:
            progress(int(100 * (ti + 1) / len(tiles)))
    band = None
    ds = None
    px_area = s["gt"][1] * abs(s["gt"][5]) / 1e4
    summary = {"mode": mode, "period": period, "model": os.path.basename(mask_model),
               "provider": model.provider, "dates": labels, "n_dates_total": len(dt_all),
               "tile_px": TILE_PX, "tiles": len(tiles),
               "changed_chips": changed_chips, "changed_px": changed_chips * CHIP * CHIP,
               "invalid_px": invalid_px,
               "changed_ha": float(changed_chips * CHIP * CHIP * px_area), "mask_tif": path}
    with open(os.path.join(out_dir, "mask_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    return summary


# ---------------------------------------------------------------- step 2: types
def _mask_windows(change, chip=CHIP, block=BLOCK):
    """Chip-aligned sub-windows covering all changed pixels of one tile.

    The tile is scanned in `block`-sized rows; changed blocks are merged into
    horizontal runs, each run is returned as its chip-aligned bounding box.
    """
    h, w = change.shape
    for by in range(0, h, block):
        band = change[by:by + block]
        cols = np.where(band.any(axis=0))[0]
        if cols.size == 0:
            continue
        bi = np.unique(cols // block)
        runs = []
        start = prev = int(bi[0])
        for k in bi[1:]:
            k = int(k)
            if k == prev + 1:
                prev = k
            else:
                runs.append((start, prev))
                start = prev = k
        runs.append((start, prev))
        for b0, b1 in runs:
            run_cols = cols[(cols >= b0 * block) & (cols < (b1 + 1) * block)]
            if run_cols.size == 0:
                continue
            sub = change[by:by + block, int(run_cols.min()):int(run_cols.max()) + 1]
            rows = np.where(sub.any(axis=1))[0]
            r0 = (by + int(rows.min())) // chip * chip
            r1 = min(h, -(-(by + int(rows.max()) + 1) // chip) * chip)
            c0 = int(run_cols.min()) // chip * chip
            c1 = min(w, -(-(int(run_cols.max()) + 1) // chip) * chip)
            yield r0, c0, r1, c1


def predict(mode, model_path, out_dir, period=None, mosaic_dir=None, files=None, t1_path=None,
            t2_path=None, band_idx=None, aoi_bbox=None, progress=None, mask_path=None,
            provider="auto"):
    """Type classification of the changed chips (tiles without changes are skipped)."""
    os.makedirs(out_dir, exist_ok=True)
    s = _setup(mode, period, mosaic_dir, files, t1_path, t2_path, aoi_bbox)
    if mode != "period":
        s["series_files"] = list(files or []) or ([t1_path, t2_path] if t1_path else [])
    gt, proj, c0, r0, w, h = s["gt"], s["proj"], s["c0"], s["r0"], s["w"], s["h"]
    cls_path = os.path.join(out_dir, "classes.tif")
    ds_out = _create_raster(cls_path, gt, c0, r0, w, h, proj)
    band = ds_out.GetRasterBand(1)

    mask_file = mask_path or os.path.join(out_dir, "mask.tif")
    use_mask = os.path.exists(mask_file)
    # the mask may be AOI-sized (written by compute_mask) or cover the whole source
    # grid: compute the AOI offset inside the mask raster from the geotransforms
    mask_dx = mask_dy = 0
    if use_mask:
        mgt, _, _, _ = _grid_from(mask_file)
        mask_dx = int(round((gt[0] + c0 * gt[1] - mgt[0]) / gt[1]))
        mask_dy = int(round((gt[3] + r0 * gt[5] - mgt[3]) / gt[5]))
    enc_path = os.path.join(os.path.dirname(model_path), "typenet_encoder.onnx")
    if not os.path.exists(enc_path):
        raise ValueError(f"Энкодер типов не найден рядом с моделью: {enc_path}")
    labels, dt_all, kind, paths = _series_for(s, mode, None)
    sel = _select_dates(len(labels))
    labels = [labels[i] for i in sel]
    dt_sel = np.zeros(len(sel), np.float32)
    for k in range(1, len(sel)):
        dt_sel[k] = float(dt_all[sel[k - 1] + 1:sel[k] + 1].sum())
    doy_sel = np.stack([_doy3(l, year=(kind != "mosaic")) for l in labels])
    enc = EncoderModel(enc_path, provider)
    cfc = TypeCfCModel(model_path, provider)
    print(f"[forest_change] типы: TypeNet (энкодер {os.path.basename(enc_path)} + CfC "
          f"{os.path.basename(model_path)}), {len(labels)} дат", flush=True)
    cells = []
    tiles = _tiles(w, h)
    if progress:
        progress(1)
    buf_chips, buf_dt, buf_doy, buf_pos = [], [], [], []

    def flush():
        if not buf_chips:
            return
        chips = np.stack(buf_chips)                       # (M, D, 6, 32, 32)
        M, D = chips.shape[0], chips.shape[1]
        feats = enc.run(chips.reshape(M * D, 6, 32, 32)).reshape(M, D, 768)
        logits = cfc.run(feats, np.stack(buf_dt), np.stack(buf_doy))
        probs = _softmax(np.asarray(logits, dtype=np.float32))
        for idx_cell, (rr, cc) in enumerate(buf_pos):
            k = int(logits[idx_cell].argmax()) + 1       # 0/1/2 -> logging/windfall/fire
            cells.append((rr, cc, k, float(probs[idx_cell, k - 1])))
        buf_chips.clear(); buf_dt.clear(); buf_doy.clear(); buf_pos.clear()

    def process_window(ax, ay, series, change):
        """Classify the changed chips of one read sub-window (series = per-date (6,h,w))."""
        th_sub, tw_sub = change.shape
        for r in range(0, th_sub - CHIP + 1, CHIP):
            for c in range(0, tw_sub - CHIP + 1, CHIP):
                frac = float(change[r:r + CHIP, c:c + CHIP].mean())
                if frac < CHANGE_FRAC:
                    continue
                chips = np.zeros((len(series), 6, CHIP, CHIP), np.float32)
                for li in range(len(series)):
                    chips[li] = series[li][:, r:r + CHIP, c:c + CHIP] / SCALE
                buf_chips.append(chips)
                buf_dt.append(dt_sel.copy())
                buf_doy.append(doy_sel.copy())
                buf_pos.append((ay + r, ax + c))
                if len(buf_chips) >= 16:      # 16 chips x <=8 dates = 128 encoder rows
                    flush()

    for ti, (x, y, tw, th) in enumerate(tiles):
        out = np.zeros((th, tw), np.uint8)
        n_before = len(cells)
        if use_mask:
            # read only the mask for this window: tiles without changes cost nothing
            mt = _read_band1(mask_file, mask_dx + x, mask_dy + y, tw, th)
            change = mt == 1
            invalid_tile = mt == 255
            if change.any():
                windows = _mask_windows(change) if SUBTILE else [(0, 0, th, tw)]
                for sr0, sc0, sr1, sc1 in windows:
                    series, invalid = [], None
                    for label in labels:
                        bands, scl = _series_read(mode, s, band_idx, label,
                                                  c0 + x + sc0, r0 + y + sr0,
                                                  sc1 - sc0, sr1 - sr0)
                        series.append(bands)
                        if scl is not None:
                            inv = np.isin(scl, SCL_INVALID)
                            invalid = inv if invalid is None else (invalid | inv)
                    sub = change[sr0:sr1, sc0:sc1]
                    if invalid is not None:
                        sub = sub & ~invalid
                    process_window(x + sc0, y + sr0, series, sub)
        else:
            series = []
            for label in labels:
                bands, _ = _series_read(mode, s, band_idx, label, c0 + x, r0 + y, tw, th)
                series.append(bands)
            A = series[0].astype(np.float32) / SCALE
            B = series[-1].astype(np.float32) / SCALE
            invalid = np.zeros((th, tw), bool)
            process_window(x, y, series, _gate(A, B, invalid))
        flush()
        # write the classes of the cells found in this tile
        for (rr, cc, k, _conf) in cells[n_before:]:
            out[rr - y:rr - y + CHIP, cc - x:cc - x + CHIP] = k
        out[invalid_tile if use_mask else invalid] = 255
        band.WriteArray(out, x, y)
        if progress:
            progress(int(100 * (ti + 1) / len(tiles)))
    band = None
    ds_out = None
    if hasattr(cfc, "close"):
        cfc.close()

    gpkg = os.path.join(out_dir, "changes.gpkg")
    _polygonize(cls_path, gpkg, proj)
    cells_path = os.path.join(out_dir, "cells.gpkg")
    _write_cells(cells_path, cells, gt, c0, r0, proj, period, s["date1"], s["date2"])
    n_px = {}
    for k in (1, 2, 3):
        n_px[CLASSES[k]] = float(sum(1 for _, _, kk, _ in cells if kk == k) * CHIP * CHIP
                                 * gt[1] * abs(gt[5]) / 1e4)
    summary = {"mode": mode, "period": period, "provider": cfc.provider,
               "chip": CHIP, "tile_px": TILE_PX, "scale_m": gt[1], "change_thr": CHANGE_THR,
               "cells": len(cells),
               "used_mask": os.path.basename(mask_file) if use_mask else None,
               "areas_ha": n_px,
               "classes_tif": cls_path, "changes_gpkg": gpkg, "cells_gpkg": cells_path}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    return summary


def _create_raster(path, gt, c0, r0, w, h, proj, nodata=255):
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(path, w, h, 1, gdal.GDT_Byte, options=["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform((gt[0] + c0 * gt[1], gt[1], 0, gt[3] + r0 * gt[5], 0, gt[5]))
    ds.SetProjection(proj)
    ds.GetRasterBand(1).SetNoDataValue(nodata)
    return ds


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
