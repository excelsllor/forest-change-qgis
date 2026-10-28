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

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from osgeo import gdal, ogr, osr

_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if os.path.isdir(os.path.join(_VENDOR, "onnxruntime")) and _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

gdal.UseExceptions()
gdal.SetConfigOption("GDAL_CACHEMAX", "512")   # keep decompressed 512x512 blocks hot

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
BOA_OFFSET = 0.1        # L2A BOA_ADD_OFFSET = -1000 (baseline 04.00); training raw DN kept it
_BOA_ENV = os.environ.get("FC_BOA", "auto").strip().lower()   # auto | raw | corrected
_BOA_MODE = {}          # source key -> True (add +0.1) / False (raw DN)
_BOA_AMBIG = {}         # source key -> p0.5 in the grey zone (convention unclear)
MASK_T = 8             # fixed sequence length of the exported type ONNX (zero padding)
MASK_CTX_T = 4         # fixed sequence length of the context mask ONNX (as in its training)
MASK_BATCH = 128       # chips per model call
DT_CANON = (0.0, 0.72, 1.15, 1.0)   # years between the canonical mosaic dates
DATE_RE = None         # compiled lazily in _parse_date
CHANGE_THR = 0.15  # |dNDVI| / |dNBR| threshold of the algorithmic change gate
MASK_PROB_THR = 0.24  # one-DOFA mask probability threshold (EA-tuned)
TYPE_BIAS = np.array([0.0, 1.5, 0.0], np.float32)  # type logit bias (EA post-processing)
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
        print(f"[forest_change] РїСЂРѕРІР°Р№РґРµСЂ '{preferred}' РЅРµРґРѕСЃС‚СѓРїРµРЅ (РµСЃС‚СЊ: {avail}), РёСЃРїРѕР»СЊР·СѓСЋ CPU",
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


def _texture9(chip):
    """9 texture features of one 32x32 chip (B08 gradients) for the Vozh-tuned TypeNet."""
    g = chip[3].astype(np.float32)
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) % np.pi) / np.pi
    hist = np.zeros(4, np.float32)
    for b in range(4):
        hist[b] = mag[(ang >= b / 4) & (ang < (b + 1) / 4)].sum()
    hist /= max(float(hist.sum()), 1e-6)
    ent = float(-(hist * np.log(hist + 1e-6)).sum() / np.log(4))
    aniso = float((hist.max() - hist.min()) / (hist.mean() + 1e-6))
    edge = float((mag > mag.mean() + mag.std()).mean())
    return np.array([*hist, ent, aniso, edge, float(g.std()), float(mag.mean())],
                    np.float32)


def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _boa_decide(arr):
    """Decide the radiometry convention from full-resolution samples of the source.

    arr: (bands, N) raw values. Raw L2A DN (offset not applied) has p0.5 ~0.1
    (the +1000 offset); offset-corrected physical reflectance is near 0/negative.
    The 0.06-0.09 band is ambiguous (bright corrected scenes / dark raw ones):
    we do NOT shift and flag it for the caller (FC_BOA overrides).
    Returns (fix, lo, ambiguous) or (None, lo, False) when too few pixels.
    """
    x = arr.astype(np.float32) / SCALE
    pos = x[x > 0]
    if pos.size < 1000:
        return None, 0.0, False
    lo = float(np.percentile(pos, 0.5))
    if lo < 0.06 or float(x.min()) < 0:
        return True, lo, False
    if lo <= 0.09:
        return False, lo, True
    return False, lo, False


def _boa_sample(src, key, bands):
    """Detect + cache the BOA convention for one source (mosaic date / file).

    Uses 4x4 blocks spread over the whole raster at FULL resolution (no resampling,
    so percentiles are honest; overviews/averaging would smooth the extremes).
    """
    if key in _BOA_MODE:
        return
    if _BOA_ENV in ("raw", "none", "off"):
        _BOA_MODE[key] = False
        print(f"[forest_change] СЂР°РґРёРѕРјРµС‚СЂРёСЏ {key}: FC_BOA=raw (Р±РµР· +0.1)", flush=True)
        return
    if _BOA_ENV in ("corrected", "offset", "on"):
        _BOA_MODE[key] = True
        print(f"[forest_change] СЂР°РґРёРѕРјРµС‚СЂРёСЏ {key}: FC_BOA=corrected (+0.1)", flush=True)
        return
    try:
        w, h = src.RasterXSize, src.RasterYSize
        blocks = []
        for by in range(4):
            for bx in range(4):
                x0 = int(bx * w / 4)
                y0 = int(by * h / 4)
                bw, bh = min(128, w - x0), min(128, h - y0)
                arrs = [src.GetRasterBand(int(b)).ReadAsArray(x0, y0, bw, bh)
                        for b in bands]
                blocks.append(np.stack(arrs).reshape(len(bands), -1))
        arr = np.concatenate(blocks, axis=1)
    except Exception:
        return
    fix, lo, amb = _boa_decide(arr)
    if fix is None:
        return
    _BOA_MODE[key] = fix
    if fix:
        print(f"[forest_change] СЂР°РґРёРѕРјРµС‚СЂРёСЏ {key}: РІС…РѕРґ Р±РµР· BOA offset -> +0.1 "
              f"(p0.5={lo:.3f})", flush=True)
    else:
        print(f"[forest_change] СЂР°РґРёРѕРјРµС‚СЂРёСЏ {key}: СЃС‹СЂРѕР№ L2A DN "
              f"(p0.5={lo:.3f})", flush=True)
    if amb:
        _BOA_AMBIG[key] = lo
        print(f"[forest_change] Р’РќРРњРђРќРР• {key}: p0.5={lo:.3f} РІ СЃРµСЂРѕР№ Р·РѕРЅРµ 0.06-0.09 вЂ” "
              f"РєРѕРЅРІРµРЅС†РёСЏ BOA РЅРµРѕРґРЅРѕР·РЅР°С‡РЅР°; Р·Р°РґР°Р№С‚Рµ FC_BOA=raw|corrected", flush=True)


def _boa_fix_raw(a, key):
    """Apply the cached +0.1 correction (training convention) to a raw window."""
    if _BOA_MODE.get(key):
        return a + np.float32(SCALE * BOA_OFFSET)
    return a


def _series_warning(labels, dt, mode):
    """Warn when the files-mode series leaves the training distribution."""
    if mode == "period":
        return None
    msgs = []
    pos = dt[1:][dt[1:] > 0]
    if pos.size and (float(pos.min()) < 0.5 or float(pos.max()) > 1.3):
        msgs.append(f"dt {float(pos.min()):.2f}..{float(pos.max()):.2f} Р»РµС‚ РІРЅРµ 0.5-1.3")
    parsed = [_parse_date(l) for l in labels]
    if labels and all(parsed):
        months = sorted({p[1] for p in parsed})
        if not set(months) <= {5, 6, 7, 9}:
            msgs.append(f"РјРµСЃСЏС†С‹ {months} РІРЅРµ [05,06,07,09]")
    elif labels:
        msgs.append("РґР°С‚С‹ РЅРµ СЂР°СЃРїРѕР·РЅР°РЅС‹ вЂ” СЃРµР·РѕРЅ/РёРЅС‚РµСЂРІР°Р»С‹ РЅРµ РїСЂРѕРІРµСЂРµРЅС‹")
    if msgs:
        msg = "Р’РќРРњРђРќРР• (files): " + "; ".join(msgs) + " вЂ” СЂРµР·СѓР»СЊС‚Р°С‚ РІРЅРµ СЂР°СЃРїСЂРµРґРµР»РµРЅРёСЏ РѕР±СѓС‡РµРЅРёСЏ"
        print(f"[forest_change] {msg}", flush=True)
        return msg
    return None


# ---------------------------------------------------------------- IO helpers
_DS_CACHE = {}


def _gdal_open(path):
    """Cached GDAL dataset (opening a 14k x 7k tiled TIFF costs ~2-5 ms per band)."""
    ds = _DS_CACHE.get(path)
    if ds is None:
        ds = gdal.Open(path)
        if ds is None:
            raise IOError(f"Cannot open {path}")
        _DS_CACHE[path] = ds
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
            f"{os.path.basename(path)}: Р·Р°РїСЂРѕС€РµРЅС‹ РєР°РЅР°Р»С‹ {list(band_idx)}, "
            f"РЅРѕ РІ С„Р°Р№Р»Рµ С‚РѕР»СЊРєРѕ {n} РєР°РЅР°Р»(РѕРІ).\n"
            "РЈРєР°Р¶РёС‚Рµ РєРѕСЂСЂРµРєС‚РЅС‹Рµ РЅРѕРјРµСЂР°: РґР»СЏ 6-РєР°РЅР°Р»СЊРЅРѕРіРѕ СЃС‚РµРєР° вЂ” 1,2,3,4,5,6; "
            "РґР»СЏ Sentinel-2 L2A вЂ” 2,3,4,8,11,12 (B02,B03,B04,B08,B11,B12).")
    out = np.zeros((len(band_idx), h, w), dtype=np.float32)
    for i, b in enumerate(band_idx):
        out[i] = ds.GetRasterBand(b).ReadAsArray(c0, r0, w, h)
    key = os.path.abspath(path)
    _boa_sample(ds, key, band_idx)
    del ds
    return _boa_fix_raw(out, key)


_VRT_CACHE = {}


def _mosaic_vrt(mosaic_dir, date):
    """In-memory 6-band VRT over <date>_<band>.tif (one GDAL call instead of six)."""
    key = (os.path.abspath(mosaic_dir), date)
    vrt = _VRT_CACHE.get(key)
    if vrt is not None:
        return vrt
    srcs = [os.path.join(mosaic_dir, f"{date}_{b}.tif") for b in BANDS6]
    for p in srcs:
        if not os.path.exists(p):
            raise ValueError(
                f"Р’ РїР°РїРєРµ РјРѕР·Р°РёРє РЅРµС‚ С„Р°Р№Р»Р° В«{os.path.basename(p)}В».\n"
                f"РџР°РїРєР°: {mosaic_dir}\nРћР¶РёРґР°СЋС‚СЃСЏ С„Р°Р№Р»С‹ <РґР°С‚Р°>_<РєР°РЅР°Р»>.tif РґР»СЏ РєР°РЅР°Р»РѕРІ "
                f"{', '.join(BANDS6)} Рё <РґР°С‚Р°>_SCL.tif РїРѕ РґР°С‚Р°Рј {', '.join(SEQ_DATES)}.")
    vrt = gdal.BuildVRT(f"/vsimem/fc_{abs(hash(key))}.vrt", srcs,
                        options=gdal.BuildVRTOptions(separate=True))
    _VRT_CACHE[key] = vrt
    return vrt


def _read_mosaic6(mosaic_dir, date, c0, r0, w, h):
    vrt = _mosaic_vrt(mosaic_dir, date)
    arr = vrt.ReadAsArray(c0, r0, w, h)
    if arr is None:
        raise IOError(f"РќРµ СѓРґР°Р»РѕСЃСЊ РїСЂРѕС‡РёС‚Р°С‚СЊ РѕРєРЅРѕ {w}x{h} ({date}) РІ РјРѕР·Р°РёРєРµ {mosaic_dir}")
    key = f"{os.path.basename(mosaic_dir)}/{date}"
    _boa_sample(vrt, key, list(range(1, vrt.RasterCount + 1)))
    return _boa_fix_raw(arr, key)


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
            raise ValueError("РќСѓР¶РЅРѕ РІС‹Р±СЂР°С‚СЊ РјРёРЅРёРјСѓРј 2 СЃРЅРёРјРєР° (pre Рё post) РёР· Р·Р°РіСЂСѓР¶РµРЅРЅС‹С… СЃР»РѕС‘РІ.")
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


def _select_dates(n, cap=None, keep_intermediate=True):
    """At most `cap` dates (default MASK_T), first and last always kept (evenly spaced).

    keep_intermediate=False -> only the first and last dates (fast mode).
    """
    cap = MASK_T if cap is None else cap
    if not keep_intermediate:
        return [0, n - 1] if n >= 2 else [0]
    if n <= cap:
        return list(range(n))
    idx = np.linspace(0, n - 1, cap).round().astype(int)
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


class MaskCtxModel:
    """Context mask CfC with dynamic T and explicit lengths (ONNX Loop).

    Runs on CPU: the ONNX Loop operator produces wrong results on DirectML.
    """

    def __init__(self, path, provider="cpu"):
        import onnxruntime as ort
        with open(path, "rb") as fh:
            buf = fh.read()
        self.sess = ort.InferenceSession(buf, providers=["CPUExecutionProvider"])

    @property
    def provider(self):
        return self.sess.get_providers()[0]

    def run(self, feats, ctx, valid, dt, doy):
        """feats (M,T,768), ctx (M,T,3,3,768), valid (M,T,3,3) -> (M,2)."""
        return self.sess.run(None, {"feats": feats.astype(np.float32),
                                    "ctx": ctx.astype(np.float32),
                                    "valid": np.asarray(valid, dtype=bool),
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
    """TypeNet CfC with dynamic T (ONNX Loop): feats + dt + doy -> (M,3) logits.

    Runs on CPU: the ONNX Loop operator produces wrong results on DirectML.
    """

    def __init__(self, path, provider="cpu"):
        import onnxruntime as ort
        with open(path, "rb") as fh:
            buf = fh.read()
        self.sess = ort.InferenceSession(buf, providers=["CPUExecutionProvider"])
        self.has_ctx = any(i.name == "ctx" for i in self.sess.get_inputs())
        self.has_tex = any(i.name == "tex" for i in self.sess.get_inputs())
        self.has_spec = any(i.name == "spec" for i in self.sess.get_inputs())

    @property
    def provider(self):
        return self.sess.get_providers()[0]

    def run(self, feats, ctx, valid, dt, doy, tex, spec):
        feed = {"feats": feats.astype(np.float32),
                "dt": dt.astype(np.float32),
                "doy": doy.astype(np.float32)}
        if self.has_ctx:
            feed["ctx"] = ctx.astype(np.float32)
            feed["valid"] = np.asarray(valid, dtype=bool)
        if self.has_tex:
            feed["tex"] = tex.astype(np.float32)
        if self.has_spec:
            feed["spec"] = spec.astype(np.float32)
        return self.sess.run(None, feed)[0]


def compute_mask(mode, out_dir, period=None, mosaic_dir=None, files=None, t1_path=None,
                 t2_path=None, band_idx=None, aoi_bbox=None, progress=None, thr=None,
                 provider="auto", mask_model=None, dates=None, use_intermediate=True,
                 force_mask=False):
    """Model-based change mask -> mask.tif (1 changed chip / 0 / 255).

    MaskNet-v2 (bicfcpn_big ONNX) processes the raw-reflectance time series of every
    32 px chip (up to 8 dates, zero padded) with dt (years) and doy (sin/cos/year).
    """
    if not mask_model or not os.path.exists(mask_model):
        raise ValueError(f"РњРѕРґРµР»СЊ РјР°СЃРєРё РЅРµ РЅР°Р№РґРµРЅР°: {mask_model}")
    _BOA_MODE.clear()
    _BOA_AMBIG.clear()
    os.makedirs(out_dir, exist_ok=True)
    s = _setup(mode, period, mosaic_dir, files, t1_path, t2_path, aoi_bbox)
    if mode != "period":
        s["series_files"] = list(files or []) or ([t1_path, t2_path] if t1_path else [])
    labels, dt_all, kind, paths = _series_for(s, mode, dates)
    warn = _series_warning(labels, dt_all, mode)
    sel = _select_dates(len(labels), cap=MASK_CTX_T, keep_intermediate=use_intermediate)
    labels = [labels[i] for i in sel]
    dt_sel = np.zeros(len(sel), np.float32)
    for k in range(1, len(sel)):
        dt_sel[k] = float(dt_all[sel[k - 1] + 1:sel[k] + 1].sum())
    doy_sel = np.stack([_doy3(l, year=(kind != "mosaic")) for l in labels]) \
        if labels else np.zeros((0, 3), np.float32)
    print(f"[forest_change] РјР°СЃРєР°: {len(sel)} РґР°С‚ РёР· {len(dt_all)} (РїРµСЂРІР°СЏ/РїРѕСЃР»РµРґРЅСЏСЏ СЃРѕС…СЂР°РЅРµРЅС‹), "
          f"РјРѕРґРµР»СЊ {os.path.basename(mask_model)}", flush=True)

    enc_path = os.path.join(os.path.dirname(mask_model), "typenet_encoder.onnx")
    if not os.path.exists(enc_path):
        raise ValueError(f"DOFA-СЌРЅРєРѕРґРµСЂ РЅРµ РЅР°Р№РґРµРЅ СЂСЏРґРѕРј СЃ РјР°СЃРєРѕР№: {enc_path}")
    src_sig = []
    if mode == "period":
        for l in labels:
            for f in (f"{l}_B04.tif", f"{l}_SCL.tif"):
                p = os.path.join(mosaic_dir, f)
                if os.path.exists(p):
                    src_sig.append((f, os.path.getmtime(p)))
    else:
        for pth in (s.get("series_files") or []):
            if pth and os.path.exists(pth):
                src_sig.append((os.path.basename(pth), os.path.getmtime(pth)))
    cache_key = hashlib.md5(json.dumps({
        "v": 4, "mode": mode, "period": period, "labels": labels,
        "dt": [round(float(x), 4) for x in dt_sel],
        "inter": bool(use_intermediate),
        "thr": float(thr if thr is not None else CHANGE_THR),
        "mask_thr": float(MASK_PROB_THR),
        "model": os.path.getmtime(mask_model), "enc": os.path.getmtime(enc_path),
        "src": src_sig, "aoi": [s["c0"], s["r0"], s["w"], s["h"]],
        "band_idx": list(band_idx) if band_idx else None,
    }, sort_keys=True).encode()).hexdigest()

    path = os.path.join(out_dir, "mask.tif")
    cache_file = os.path.join(out_dir, "mask_cache.json")
    if not force_mask and os.path.exists(path):
        try:
            with open(cache_file, encoding="utf-8") as fh:
                cached = json.load(fh)
            if cached.get("key") == cache_key:
                summary = cached["summary"]
                summary["cached"] = True
                print(f"[forest_change] РјР°СЃРєР°: РєСЌС€ ({summary.get('changed_ha', 0):.1f} РіР°) вЂ” "
                      f"РїСЂРѕРїСѓСЃРєР°СЋ СЂР°СЃС‡С‘С‚", flush=True)
                return summary
        except Exception:
            pass

    ds = _create_raster(path, s["gt"], s["c0"], s["r0"], s["w"], s["h"], s["proj"])
    band = ds.GetRasterBand(1)
    tiles = _tiles(s["w"], s["h"])
    enc = EncoderModel(enc_path, provider)
    model = MaskCtxModel(mask_model, provider)
    print(f"[forest_change] РјР°СЃРєР°: СЌРЅРєРѕРґРµСЂ {os.path.basename(enc_path)} + attention-РєРѕРЅС‚РµРєСЃС‚ 3x3 "
          f"+ CfC {os.path.basename(mask_model)}", flush=True)
    changed_chips = 0
    invalid_px = 0
    if progress:
        progress(1)
    buf = []
    HALO = CHIP

    def flush():
        for feats, ctx, cval, dtb, doyb, pos in buf:
            n = len(feats)
            for i0 in range(0, n, 512):
                logits = model.run(feats[i0:i0 + 512], ctx[i0:i0 + 512],
                                   cval[i0:i0 + 512], dtb[i0:i0 + 512],
                                   doyb[i0:i0 + 512])
                logits = np.asarray(logits, dtype=np.float32).reshape(-1)
                probs = 1.0 / (1.0 + np.exp(-logits))
                for j in range(probs.shape[0]):
                    if float(probs[j]) > MASK_PROB_THR:
                        cells_changed.append(pos[i0 + j])
        buf.clear()

    cells_changed = []
    ex = ThreadPoolExecutor(max_workers=1)

    def read_series_box(x, y, tw, th):
        """Read the extended series window (halo for the 3x3 context)."""
        hx0 = x - HALO if x >= HALO else 0
        hy0 = y - HALO if y >= HALO else 0
        hx1 = min(s["w"], x + tw + HALO)
        hy1 = min(s["h"], y + th + HALO)
        ew, eh = hx1 - hx0, hy1 - hy0
        reads = []
        for label in labels:
            bands, scl = _series_read(mode, s, band_idx, label,
                                      s["c0"] + hx0, s["r0"] + hy0, ew, eh)
            reads.append((bands, scl))
        return hx0, hy0, ew, eh, reads

    def process_tile(ti, x, y, tw, th, fut):
        nonlocal changed_chips, invalid_px
        hx0, hy0, ew, eh, reads = fut.result()
        out = np.zeros((th, tw), np.uint8)
        invalid = np.zeros((th, tw), bool)
        nc, nr = ew // CHIP, eh // CHIP
        ox, oy = x - hx0, y - hy0
        grid = np.zeros((len(labels), nr, nc, 768), np.float32)
        gvalid = np.zeros((len(labels), nr, nc), bool)
        series = []
        for li, (bands, scl) in enumerate(reads):
            series.append(bands)
            if scl is not None and li in (0, len(labels) - 1):
                invalid |= np.isin(scl[oy:oy + th, ox:ox + tw], SCL_INVALID)
            b = bands[:, :nr * CHIP, :nc * CHIP].astype(np.float32) / SCALE
            chips = b.reshape(6, nr, CHIP, nc, CHIP).transpose(1, 3, 0, 2, 4)
            chips = np.ascontiguousarray(chips).reshape(nr * nc, 6, CHIP, CHIP)
            gvalid[li] = (chips.sum(axis=(1, 2, 3)) > 0).reshape(nr, nc)
            f = np.empty((nr * nc, 768), np.float32)
            for i0 in range(0, nr * nc, 512):
                f[i0:i0 + 512] = enc.run(chips[i0:i0 + 512])
            grid[li] = f.reshape(nr, nc, 768)
        first_i = series[0][:, oy:oy + th, ox:ox + tw]
        last_i = series[-1][:, oy:oy + th, ox:ox + tw]
        invalid |= (first_i[3] == 0) | (last_i[3] == 0)
        if FOREST_NDVI_MIN > 0:
            invalid |= (_ndvi(first_i) < FOREST_NDVI_MIN) & (_ndvi(last_i) < FOREST_NDVI_MIN)
        cells_changed.clear()
        nh, nw = th // CHIP, tw // CHIP
        if nh and nw:
            inv_frac = invalid[:nh * CHIP, :nw * CHIP].reshape(
                nh, CHIP, nw, CHIP).mean(axis=(1, 3))
            sel = np.argwhere(inv_frac <= 0.5)
            if len(sel):
                n = len(sel)
                Tn = len(labels)
                g0 = (y - hy0) // CHIP
                gg0 = (x - hx0) // CHIP
                rows = g0 + sel[:, 0]
                cols = gg0 + sel[:, 1]
                pad = np.zeros((Tn, nr + 2, nc + 2, 768), np.float32)
                pad[:, 1:-1, 1:-1] = grid
                pv = np.zeros((Tn, nr + 2, nc + 2), bool)
                pv[:, 1:-1, 1:-1] = gvalid
                feats = grid[:, rows, cols].transpose(1, 0, 2)          # (n,T,256)
                ctx = np.empty((n, Tn, 3, 3, 768), np.float32)
                cval = np.empty((n, Tn, 3, 3), bool)
                for dj in (-1, 0, 1):
                    for di in (-1, 0, 1):
                        ctx[:, :, dj + 1, di + 1] = pad[:, rows + dj + 1,
                                                        cols + di + 1].transpose(1, 0, 2)
                        cval[:, :, dj + 1, di + 1] = pv[:, rows + dj + 1,
                                                         cols + di + 1].transpose(1, 0)
                buf.append((feats, ctx, cval, np.tile(dt_sel, (n, 1)),
                            np.tile(doy_sel, (n, 1, 1)),
                            [(int(j) * CHIP, int(i) * CHIP) for j, i in sel]))
        flush()
        for (r, c) in cells_changed:
            out[r:r + CHIP, c:c + CHIP] = 1
            changed_chips += 1
        out[invalid] = 255
        band.WriteArray(out, x, y)
        invalid_px += int(invalid.sum())
        if progress:
            progress(int(100 * (ti + 1) / len(tiles)))

    prev = None
    for ti, (x, y, tw, th) in enumerate(tiles):
        fut = ex.submit(read_series_box, x, y, tw, th)
        if prev is not None:
            process_tile(*prev)
        prev = (ti, x, y, tw, th, fut)
    if prev is not None:
        process_tile(*prev)
    ex.shutdown()
    band = None
    ds = None
    px_area = s["gt"][1] * abs(s["gt"][5]) / 1e4
    summary = {"mode": mode, "period": period, "model": os.path.basename(mask_model),
               "provider": enc.provider, "cfc_provider": model.provider, "dates": labels, "n_dates_total": len(dt_all),
               "boa_offset": sorted(str(k) for k, v in _BOA_MODE.items() if v),
               "boa_ambiguous": sorted(f"{k} p0.5={v:.3f}" for k, v in _BOA_AMBIG.items()),
               "boa_checked": len(_BOA_MODE), "boa_mode": _BOA_ENV,
               "series_warning": warn, "intermediate_mask": bool(use_intermediate),
               "cached": False, "thr": float(thr if thr is not None else CHANGE_THR),
               "mask_thr": float(MASK_PROB_THR),
               "tile_px": TILE_PX, "tiles": len(tiles),
               "changed_chips": changed_chips, "changed_px": changed_chips * CHIP * CHIP,
               "invalid_px": invalid_px,
               "changed_ha": float(changed_chips * CHIP * CHIP * px_area), "mask_tif": path}
    with open(os.path.join(out_dir, "mask_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    with open(cache_file, "w", encoding="utf-8") as fh:
        json.dump({"key": cache_key, "summary": summary}, fh, ensure_ascii=False)
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


def compute_disturbance(mode, out_dir, period=None, mosaic_dir=None, files=None, t1_path=None,
                        t2_path=None, band_idx=None, aoi_bbox=None, progress=None,
                        mask_path=None, preset="standard"):
    """Pixel-level disturbance mask inside the model-changed chips (dNDVI/dNBR).

    Third mask for an approximate affected area: for the first/last dates of the series,
    flags 10 m pixels with dNDVI < -th_d or dNBR > th_b (and SCL/forest validity) inside
    the model mask (chip-level). Output disturbance.tif: 1 disturbed, 0 changed-but-valid,
    255 outside the model mask / invalid. Presets: standard (0.10/0.05), conservative
    (0.15/0.10), aggressive (0.20/0.15); area = px * 0.01 ha.
    """
    _BOA_MODE.clear()
    _BOA_AMBIG.clear()
    os.makedirs(out_dir, exist_ok=True)
    s = _setup(mode, period, mosaic_dir, files, t1_path, t2_path, aoi_bbox)
    if mode != "period":
        s["series_files"] = list(files or []) or ([t1_path, t2_path] if t1_path else [])
    labels, _dt_all, _kind, _paths = _series_for(s, mode, None)
    if len(labels) < 2:
        raise ValueError("Р”Р»СЏ РїРёРєСЃРµР»СЊРЅРѕР№ РјР°СЃРєРё РЅСѓР¶РЅРѕ РјРёРЅРёРјСѓРј 2 РґР°С‚С‹")
    if mode == "period":
        pre_l = s.get("date1") or labels[0]
        post_l = s.get("date2") or labels[-1]
        if pre_l not in labels or post_l not in labels:
            pre_l, post_l = labels[0], labels[-1]
    else:
        pre_l, post_l = labels[0], labels[-1]
    if preset == "conservative":
        th_d, th_b = 0.15, 0.10
    elif preset == "aggressive":
        th_d, th_b = 0.20, 0.15
    else:
        th_d, th_b = 0.10, 0.05
    mask_file = mask_path or os.path.join(out_dir, "mask.tif")
    if not os.path.exists(mask_file):
        raise ValueError(f"РњР°СЃРєР° РЅРµ РЅР°Р№РґРµРЅР°: {mask_file}")
    mgt, _, _, _ = _grid_from(mask_file)
    mask_dx = int(round((s["gt"][0] + s["c0"] * s["gt"][1] - mgt[0]) / s["gt"][1]))
    mask_dy = int(round((s["gt"][3] + s["r0"] * s["gt"][5] - mgt[3]) / s["gt"][5]))
    path = os.path.join(out_dir, "disturbance.tif")
    ds = _create_raster(path, s["gt"], s["c0"], s["r0"], s["w"], s["h"], s["proj"])
    band = ds.GetRasterBand(1)
    tiles = _tiles(s["w"], s["h"])
    dist_px = chip_px = 0
    for ti, (x, y, tw, th) in enumerate(tiles):
        pre_b, pre_scl = _series_read(mode, s, band_idx, pre_l, s["c0"] + x, s["r0"] + y, tw, th)
        post_b, post_scl = _series_read(mode, s, band_idx, post_l, s["c0"] + x, s["r0"] + y, tw, th)
        m = _read_band1(mask_file, mask_dx + x, mask_dy + y, tw, th)
        b1 = pre_b.astype(np.float32) / SCALE
        b2 = post_b.astype(np.float32) / SCALE
        nd1 = (b1[3] - b1[2]) / np.maximum(b1[3] + b1[2], 1e-6)
        nd2 = (b2[3] - b2[2]) / np.maximum(b2[3] + b2[2], 1e-6)
        nb1 = (b1[3] - b1[5]) / np.maximum(b1[3] + b1[5], 1e-6)
        nb2 = (b2[3] - b2[5]) / np.maximum(b2[3] + b2[5], 1e-6)
        valid = np.ones((th, tw), bool)
        if pre_scl is not None:
            valid &= ~np.isin(pre_scl, SCL_INVALID)
        if post_scl is not None:
            valid &= ~np.isin(post_scl, SCL_INVALID)
        valid &= (nd1 > FOREST_NDVI_MIN) | (nd2 > FOREST_NDVI_MIN)
        changed = m == 1
        dist = valid & changed & (((nd2 - nd1) < -th_d) | ((nb1 - nb2) > th_b))
        out = np.full((th, tw), 255, np.uint8)
        out[valid & changed] = 0
        out[dist] = 1
        band.WriteArray(out, x, y)
        dist_px += int(dist.sum())
        chip_px += int(changed.sum())
        if progress:
            progress(int(100 * (ti + 1) / len(tiles)))
    band = None
    ds = None
    summary = {"disturbance_tif": path, "preset": preset,
               "th_dndvi": th_d, "th_dnbr": th_b, "dates": [pre_l, post_l],
               "disturbance_px": dist_px, "disturbance_ha": dist_px * 0.01,
               "chip_ha": chip_px * 0.01,
               "ratio_to_chips": dist_px / max(chip_px, 1)}
    with open(os.path.join(out_dir, "disturbance_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"[forest_change] РїР»РѕС‰Р°РґСЊ: {summary['disturbance_ha']:.1f} РіР° "
          f"({preset}: dNDVI<-{th_d}, dNBR>{th_b}) РІРЅСѓС‚СЂРё {summary['chip_ha']:.1f} РіР° С‡РёРїРѕРІ",
          flush=True)
    return summary


def predict(mode, model_path, out_dir, period=None, mosaic_dir=None, files=None, t1_path=None,
            t2_path=None, band_idx=None, aoi_bbox=None, progress=None, mask_path=None,
            provider="auto", use_intermediate=True):
    """Type classification of the changed chips (tiles without changes are skipped)."""
    _BOA_MODE.clear()
    _BOA_AMBIG.clear()
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
        raise ValueError(f"Р­РЅРєРѕРґРµСЂ С‚РёРїРѕРІ РЅРµ РЅР°Р№РґРµРЅ СЂСЏРґРѕРј СЃ РјРѕРґРµР»СЊСЋ: {enc_path}")
    labels, dt_all, kind, paths = _series_for(s, mode, None)
    warn = _series_warning(labels, dt_all, mode)
    sel = _select_dates(len(labels), keep_intermediate=use_intermediate)
    labels = [labels[i] for i in sel]
    dt_sel = np.zeros(len(sel), np.float32)
    for k in range(1, len(sel)):
        dt_sel[k] = float(dt_all[sel[k - 1] + 1:sel[k] + 1].sum())
    doy_sel = np.stack([_doy3(l, year=(kind != "mosaic")) for l in labels])
    enc = EncoderModel(enc_path, provider)
    cfc = TypeCfCModel(model_path, provider)
    _sig_src = (os.path.join(mosaic_dir, f"{labels[0]}_B04.tif") if mode == "period"
                else (paths[0] if paths else ""))
    _sig = os.path.getmtime(_sig_src) if _sig_src and os.path.exists(_sig_src) else 0
    cache_key = hashlib.md5(
        (os.path.abspath(model_path) + "|" + str(os.path.getmtime(model_path)) + "|" +
         os.path.abspath(enc_path) + "|" + str(os.path.getmtime(enc_path)) + "|" +
         str(_sig) + "|" + ",".join(map(str, labels))).encode()).hexdigest()[:16]
    cache_dir = os.path.join(out_dir, "cache")
    cache_path = os.path.join(cache_dir, f"type_feats_{cache_key}.npz")
    fcache = {}
    if os.path.exists(cache_path):
        _zc = np.load(cache_path)
        _zf = _zc["feats"][:]
        _zk = _zc["keys"][:]
        fcache = {(int(a), int(b)): _zf[i] for i, (a, b) in enumerate(_zk)}
        print(f"[forest_change] РєСЌС€ С„РёС‡ С‚РёРїРѕРІ: {len(fcache)} С‡РёРїРѕРІ "
              f"({os.path.basename(cache_path)})", flush=True)
    print(f"[forest_change] С‚РёРїС‹: TypeNet (СЌРЅРєРѕРґРµСЂ {os.path.basename(enc_path)} + "
          f"cross-attention readout {os.path.basename(model_path)}, РґРёРЅР°РјРёС‡РµСЃРєРѕРµ T, "
          f"readout РЅР° CPU{', texture 9' if cfc.has_spec else ''}), {len(labels)} РґР°С‚",
          flush=True)
    cells = []
    tiles = _tiles(w, h)
    if progress:
        progress(1)
    pend = []
    pend_n = [0]

    def flush():
        if not pend:
            return
        chips = pend[0][0] if len(pend) == 1 else np.concatenate([p[0] for p in pend], 0)
        ctxc = pend[0][1] if len(pend) == 1 else np.concatenate([p[1] for p in pend], 0)
        dtb = pend[0][2] if len(pend) == 1 else np.concatenate([p[2] for p in pend], 0)
        doyb = pend[0][3] if len(pend) == 1 else np.concatenate([p[3] for p in pend], 0)
        pos = [q for p in pend for q in p[4]]
        M, D = chips.shape[0], chips.shape[1]
        keys = [(int(r0 + rr), int(c0 + cc)) for (rr, cc) in pos]
        feats = np.empty((M, D, 768), np.float32)
        miss = [i for i, k_ in enumerate(keys) if k_ not in fcache]
        if miss:
            flat = chips[miss].reshape(len(miss) * D, 6, CHIP, CHIP)
            mf = np.empty((len(miss) * D, 768), np.float32)
            for i0 in range(0, len(miss) * D, 512):
                mf[i0:i0 + 512] = enc.run(flat[i0:i0 + 512])
            mf = mf.reshape(len(miss), D, 768)
            for j, mi in enumerate(miss):
                feats[mi] = mf[j]
                fcache[keys[mi]] = mf[j].astype(np.float16)
        for i, k_ in enumerate(keys):
            if k_ not in miss and i not in miss:
                feats[i] = fcache[k_]
        C = ctxc.shape[1]
        cf = np.empty((M, C, D, 768), np.float32)
        flatc = ctxc.reshape(M * C * D, 6, CHIP, CHIP)
        for i0 in range(0, M * C * D, 512):
            cf.reshape(-1, 768)[i0:i0 + 512] = enc.run(flatc[i0:i0 + 512])
        ctx = cf.reshape(M, 3, 3, D, 768).transpose(0, 3, 1, 2, 4)
        valid = np.ones((M, D, 3, 3), np.float32)
        texb = np.empty((M, D, 9), np.float32)
        specb = np.empty((M, D, 9), np.float32)
        for i in range(M):
            for d in range(D):
                texb[i, d] = _texture9(np.clip(chips[i, d], 0.0, 1.5))
                specb[i, d] = _features9(chips[i, d] * SCALE).mean(axis=(1, 2))
        for i0 in range(0, M, 512):
            logits = cfc.run(feats[i0:i0 + 512], ctx[i0:i0 + 512],
                             valid[i0:i0 + 512], dtb[i0:i0 + 512],
                             doyb[i0:i0 + 512], texb[i0:i0 + 512],
                             specb[i0:i0 + 512])
            logits = np.asarray(logits, dtype=np.float32) + TYPE_BIAS
            probs = _softmax(logits)
            for j in range(logits.shape[0]):
                k = int(logits[j].argmax()) + 1       # 0/1/2 -> logging/windfall/fire
                rr, cc = pos[i0 + j]
                cells.append((rr, cc, k, float(probs[j, k - 1])))
        pend.clear(); pend_n[0] = 0

    def process_window(ax, ay, series, change):
        """Classify the changed chips of one read sub-window (series = per-date (6,h,w))."""
        D = len(series)
        th_sub, tw_sub = change.shape
        nh, nw = th_sub // CHIP, tw_sub // CHIP
        if nh == 0 or nw == 0:
            return
        frac = change[:nh * CHIP, :nw * CHIP].reshape(nh, CHIP, nw, CHIP).mean(axis=(1, 3))
        sel = np.argwhere(frac >= CHANGE_FRAC)
        if len(sel) == 0:
            return
        chips = np.empty((len(sel), D, 6, CHIP, CHIP), np.float32)
        ctxc = np.empty((len(sel), 9, D, 6, CHIP, CHIP), np.float32)
        for li in range(D):
            b = series[li][:, :nh * CHIP, :nw * CHIP].astype(np.float32) / SCALE
            arr = b.reshape(6, nh, CHIP, nw, CHIP).transpose(1, 3, 0, 2, 4)
            chips[:, li] = arr[sel[:, 0], sel[:, 1]]
        for n, (i, j) in enumerate(sel):
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    ii = min(max(int(i) + dj, 0), nh - 1)
                    jj = min(max(int(j) + di, 0), nw - 1)
                    k = (dj + 1) * 3 + (di + 1)
                    for li in range(D):
                        ctxc[n, k, li] = series[li][
                            :, ii * CHIP:(ii + 1) * CHIP,
                            jj * CHIP:(jj + 1) * CHIP].astype(np.float32) / SCALE
        pend.append((chips, ctxc, np.tile(dt_sel, (len(sel), 1)),
                     np.tile(doy_sel, (len(sel), 1, 1)),
                     [(ay + int(j) * CHIP, ax + int(i) * CHIP) for j, i in sel]))
        pend_n[0] += len(sel)
        if pend_n[0] >= 16:
            flush()

    ex = ThreadPoolExecutor(max_workers=1)

    def read_window(x, y, w):
        sr0, sc0, sr1, sc1 = w
        reads = []
        for label in labels:
            bands, scl = _series_read(mode, s, band_idx, label,
                                      c0 + x + sc0, r0 + y + sr0,
                                      sc1 - sc0, sr1 - sr0)
            reads.append((bands, scl))
        return reads

    for ti, (x, y, tw, th) in enumerate(tiles):
        out = np.zeros((th, tw), np.uint8)
        n_before = len(cells)
        if use_mask:
            # read only the mask for this window: tiles without changes cost nothing
            mt = _read_band1(mask_file, mask_dx + x, mask_dy + y, tw, th)
            change = mt == 1
            invalid_tile = mt == 255
            if change.any():
                windows = list(_mask_windows(change) if SUBTILE else [(0, 0, th, tw)])
                fut = ex.submit(read_window, x, y, windows[0]) if windows else None
                for wi, (sr0, sc0, sr1, sc1) in enumerate(windows):
                    reads = fut.result()
                    if wi + 1 < len(windows):
                        fut = ex.submit(read_window, x, y, windows[wi + 1])
                    series, invalid = [], None
                    for bands, scl in reads:
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
    ex.shutdown()
    if fcache:
        os.makedirs(cache_dir, exist_ok=True)
        _ck = np.array(list(fcache.keys()), np.int32)
        _cf = np.stack([fcache[k_] for k_ in fcache]).astype(np.float16)
        np.savez(cache_path, keys=_ck, feats=_cf)
        print(f"[forest_change] РєСЌС€ С„РёС‡ С‚РёРїРѕРІ СЃРѕС…СЂР°РЅС‘РЅ: {len(fcache)} С‡РёРїРѕРІ "
              f"({os.path.getsize(cache_path) / 1e6:.0f} РњР‘)", flush=True)
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
    summary = {"mode": mode, "period": period, "provider": enc.provider,
               "cfc_provider": cfc.provider,
               "boa_offset": sorted(str(k) for k, v in _BOA_MODE.items() if v),
               "boa_ambiguous": sorted(f"{k} p0.5={v:.3f}" for k, v in _BOA_AMBIG.items()),
               "boa_checked": len(_BOA_MODE), "boa_mode": _BOA_ENV,
               "series_warning": warn, "intermediate_types": bool(use_intermediate),
               "chip": CHIP, "tile_px": TILE_PX, "scale_m": gt[1], "change_thr": CHANGE_THR,
               "type_bias": [float(x) for x in TYPE_BIAS],
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

