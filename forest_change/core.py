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
# The bundled onnxruntime is a Windows build (DML/CPU); on other platforms the
# system onnxruntime must be used instead, otherwise the vendor package shadows
# it and the .pyd import fails.
if sys.platform == "win32" and os.path.isdir(os.path.join(_VENDOR, "onnxruntime")) \
        and _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

gdal.UseExceptions()
gdal.SetConfigOption("GDAL_CACHEMAX", os.environ.get("FC_GDAL_CACHE", "256"))

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
    "openvino": ["CPUExecutionProvider"],
    "ov": ["CPUExecutionProvider"],
    "openvino-bf16": ["CPUExecutionProvider"],
    "ov-bf16": ["CPUExecutionProvider"],
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
UNET_THR = 0.438  # CfC U-Net pixel threshold (EA)
UNET_MIN_PX = 11  # remove components smaller than this (EA)
GATE_MODE = os.environ.get("FC_GATE", "fixed").lower()   # fixed | adaptive | both
GATE_Z = float(os.environ.get("FC_GATE_Z", "4.0"))       # robust z-score threshold (adaptive)
CHANGE_FRAC = 0.02 # chip is changed if at least this fraction of pixels changed
MIN_AREA_HA = 0.0  # drop merged polygons smaller than this (0 = keep all)
TILE_PX = int(os.environ.get("FC_TILE", "1024"))   # processing window (multiple of CHIP)
SUBTILE = os.environ.get("FC_SUBTILE", "1") == "1"  # read only the mask bbox inside a tile
BLOCK = max(CHIP, int(os.environ.get("FC_BLOCK", "128")) // CHIP * CHIP)  # mask block for reads
RUN_BATCH = max(1, int(os.environ.get("FC_BATCH", "64")))          # encoder chips per call
CFC_BATCH = max(1, int(os.environ.get("FC_CFC_BATCH", "128")))     # CfC chips per call
THREADS = max(1, int(os.environ.get("FC_THREADS",
                                    str(max(1, (os.cpu_count() or 8) // 2)))))
OV_BATCH = max(1, int(os.environ.get("FC_OV_BATCH", str(RUN_BATCH))))
OV_PRECISION = os.environ.get("FC_OV_PRECISION", "f32").strip().lower()
if OV_PRECISION not in ("f32", "bf16"):
    OV_PRECISION = "f32"
FEATURE_STORE = os.environ.get("FC_FEATURES", "1").strip().lower() not in (
    "0", "false", "off", "no")
FEATURE_DTYPE = os.environ.get("FC_FEATURE_DTYPE", "f32").strip().lower()
FEATURE_DIR = os.environ.get("FC_FEATURE_DIR", "").strip()
FEATURE_MAX_MB = float(os.environ.get("FC_FEATURE_MAX_MB", "0") or 0)
FC_COG = os.environ.get("FC_COG", "1").strip().lower() not in ("0", "false", "off", "no")
FC_COG_FULL = os.environ.get("FC_COG_FULL", "0").strip().lower() not in (
    "0", "false", "off", "no")
COG_BLOCK = max(128, int(os.environ.get("FC_COG_BLOCK", "512")))


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


def _ort_options():
    """Bounded CPU session: physical-core threads, no growing arena, no copies."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = True
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return so


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
    """Cached GDAL dataset, reopened when the file mtime changes."""
    entry = _DS_CACHE.get(path)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None
    if entry is not None and entry[1] == mtime:
        return entry[0]
    ds = gdal.Open(path)
    if ds is None:
        raise IOError(f"Cannot open {path}")
    _DS_CACHE[path] = (ds, mtime)
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
    interleave = ds.GetMetadataItem("INTERLEAVE")
    if str(interleave or "").upper() == "PIXEL" and len(band_idx) > 1:
        arr = ds.ReadAsArray(c0, r0, w, h)
        if arr is None:
            del ds
            raise IOError(f"Cannot read {w}x{h} window from {path}")
        if arr.ndim == 2:
            arr = arr[None]
        out = np.ascontiguousarray(arr[[int(b) - 1 for b in band_idx]],
                                   dtype=np.float32)
    else:
        out = np.zeros((len(band_idx), h, w), dtype=np.float32)
        for i, b in enumerate(band_idx):
            out[i] = ds.GetRasterBand(b).ReadAsArray(c0, r0, w, h)
    key = os.path.abspath(path)
    _boa_sample(ds, key, band_idx)
    del ds
    return _boa_fix_raw(out, key)


def _source_sig(mode, s, labels):
    """Cheap source fingerprint: all bands/SCL mtimes for the requested dates."""
    sig = []
    if mode == "period":
        md = s.get("mosaic_dir") or ""
        for label in labels:
            for b in BANDS6:
                p = os.path.join(md, f"{label}_{b}.tif")
                if os.path.exists(p):
                    sig.append([os.path.abspath(p), os.path.getmtime(p)])
            p = os.path.join(md, f"{label}_SCL.tif")
            if os.path.exists(p):
                sig.append([os.path.abspath(p), os.path.getmtime(p)])
    else:
        for p in (s.get("series_files") or []):
            if p and os.path.exists(p):
                sig.append([os.path.abspath(p), os.path.getmtime(p)])
    return sig


def _cog_inputs(files, aoi_bbox, out_dir):
    """Optional AOI-cropped tiled COG cache for files-mode inputs."""
    if not FC_COG or not files or (not aoi_bbox and not FC_COG_FULL):
        return files, aoi_bbox
    try:
        ref = files[0]
        gt, W, H, _proj = _grid_from(ref)
        if aoi_bbox:
            xmin, ymin, xmax, ymax = aoi_bbox
            c0 = max(0, int((xmin - gt[0]) / gt[1]))
            r0 = max(0, int((ymax - gt[3]) / gt[5]))
            c1 = min(W, int((xmax - gt[0]) / gt[1]))
            r1 = min(H, int((ymin - gt[3]) / gt[5]))
        else:
            c0, r0, c1, r1 = 0, 0, W, H
        w, h = c1 - c0, r1 - r0
        if w <= 0 or h <= 0:
            return files, aoi_bbox
        ds = _gdal_open(ref)
        rb = ds.GetRasterBand(1)
        bs = rb.GetBlockSize()
        fp = rb.DataType in (gdal.GDT_Float32, gdal.GDT_Float64)
        del ds
        if c0 == 0 and r0 == 0 and w == W and h == H and bs[1] > 1 and bs[0] < W:
            return files, aoi_bbox
        cog_dir = os.path.join(out_dir, "cog")
        os.makedirs(cog_dir, exist_ok=True)
        block = max(16, min(COG_BLOCK, w, h) // 16 * 16)
        out = []
        for p in files:
            stem = os.path.splitext(os.path.basename(p))[0]
            tag = hashlib.md5(os.path.abspath(p).encode()).hexdigest()[:8]
            dst = os.path.join(cog_dir, f"{stem}_{tag}_{c0}_{r0}_{w}_{h}.tif")
            if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(p):
                tmp = dst + ".tmp.tif"
                if os.path.exists(tmp):
                    os.remove(tmp)
                gdal.Translate(tmp, p, srcWin=[c0, r0, w, h],
                               creationOptions=["TILED=YES",
                                                f"BLOCKXSIZE={block}",
                                                f"BLOCKYSIZE={block}",
                                                "COMPRESS=DEFLATE",
                                                f"PREDICTOR={3 if fp else 2}",
                                                "BIGTIFF=IF_SAFER"])
                cov = gdal.Open(tmp, gdal.GA_Update)
                cov.BuildOverviews("AVERAGE", [2, 4, 8, 16])
                cov = None
                os.replace(tmp, dst)
            out.append(dst)
        print(f"[forest_change] COG-кэш: {len(out)} файл(ов), окно {w}x{h} "
              f"({c0},{r0})", flush=True)
        return out, None
    except Exception as exc:
        print(f"[forest_change] COG-кэш недоступен ({exc}); исходные файлы",
              flush=True)
        return files, aoi_bbox


class FeatureStore:
    """Persistent per-chip encoder features shared by mask and type stages.

    Layout: (gh, gw, n_dates, 768) float32 + (gh, gw, n_dates) uint8 validity,
    indexed in AOI-relative 32 px chips. A memmap keeps RAM bounded; only touched
    tiles allocate pages.
    """

    VERSION = 2

    def __init__(self, out_dir, labels, aoi, src_sig, band_idx, mode, enabled=True,
                 model_sig=None, engine=None):
        self.enabled = bool(enabled) and bool(labels) and aoi[2] > 0 and aoi[3] > 0
        self.labels = [str(x) for x in labels]
        self.index = {label: i for i, label in enumerate(self.labels)}
        self.aoi = tuple(int(v) for v in aoi)
        self.band_idx = list(band_idx) if band_idx else None
        self.ndates = len(self.labels)
        self.gw = max(1, int(np.ceil(self.aoi[2] / CHIP)))
        self.gh = max(1, int(np.ceil(self.aoi[3] / CHIP)))
        self.dtype = np.float16 if FEATURE_DTYPE in ("f16", "float16", "half") \
            else np.float32
        self.dtype_name = "f16" if self.dtype == np.float16 else "f32"
        self.feat = None
        self.valid = None
        key = {
            "v": self.VERSION, "mode": mode, "labels": self.labels,
            "aoi": list(self.aoi), "band_idx": self.band_idx,
            "chip": CHIP, "scale": SCALE, "boa": _BOA_ENV,
            "dtype": str(np.dtype(self.dtype)), "src": src_sig,
            "model": model_sig, "engine": engine,
        }
        self.key = hashlib.md5(
            json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.dir = (os.path.join(FEATURE_DIR, self.key[:16]) if FEATURE_DIR
                    else os.path.join(out_dir, "features"))
        if not self.enabled:
            return
        try:
            os.makedirs(self.dir, exist_ok=True)
            man_path = os.path.join(self.dir, "manifest.json")
            feat_path = os.path.join(self.dir, f"features.{self.dtype_name}")
            alt_path = os.path.join(self.dir, "features.f16" if self.dtype_name == "f32"
                                    else "features.f32")
            valid_path = os.path.join(self.dir, "valid.u8")
            expected = self.gh * self.gw * self.ndates * 768 * np.dtype(self.dtype).itemsize
            expected_valid = self.gh * self.gw * self.ndates
            if FEATURE_MAX_MB > 0 and expected > FEATURE_MAX_MB * 1e6:
                raise RuntimeError(
                    f"нужно {expected / 1e6:.0f} МБ, лимит {FEATURE_MAX_MB:.0f} МБ")
            if os.path.exists(man_path) and os.path.exists(feat_path) \
                    and os.path.exists(valid_path):
                try:
                    with open(man_path, encoding="utf-8") as fh:
                        man = json.load(fh)
                except Exception:
                    man = {}
                if man.get("key") == self.key \
                        and os.path.getsize(feat_path) == expected \
                        and os.path.getsize(valid_path) == expected_valid:
                    self.feat = np.memmap(feat_path, dtype=self.dtype, mode="r+",
                                          shape=(self.gh, self.gw, self.ndates, 768))
                    self.valid = np.memmap(valid_path, dtype=np.uint8, mode="r+",
                                           shape=(self.gh, self.gw, self.ndates))
                    return
            for p in (feat_path, alt_path, valid_path):
                if os.path.exists(p):
                    os.remove(p)
            self.feat = np.memmap(feat_path, dtype=self.dtype, mode="w+",
                                  shape=(self.gh, self.gw, self.ndates, 768))
            self.valid = np.memmap(valid_path, dtype=np.uint8, mode="w+",
                                   shape=(self.gh, self.gw, self.ndates))
            with open(man_path, "w", encoding="utf-8") as fh:
                json.dump({"key": self.key, "version": self.VERSION,
                           "labels": self.labels, "aoi": list(self.aoi),
                           "band_idx": self.band_idx, "chip": CHIP,
                           "dtype": str(np.dtype(self.dtype)),
                           "shape": [self.gh, self.gw, self.ndates, 768],
                           "src": src_sig}, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[forest_change] feature store отключён: {exc}", flush=True)
            self.enabled = False
            self.feat = None
            self.valid = None

    def date_index(self, label):
        return self.index.get(str(label))

    def write_tile(self, grid, valid, gx0, gy0, date_indices):
        if not self.enabled or self.feat is None:
            return
        t_count = grid.shape[0]
        nr, nc = grid.shape[1], grid.shape[2]
        for t, di in enumerate(date_indices):
            if t >= t_count or di is None:
                continue
            y1 = min(self.gh, gy0 + nr)
            x1 = min(self.gw, gx0 + nc)
            if y1 <= gy0 or x1 <= gx0:
                continue
            self.feat[gy0:y1, gx0:x1, di, :] = np.asarray(
                grid[t, :y1 - gy0, :x1 - gx0], dtype=self.dtype)
            self.valid[gy0:y1, gx0:x1, di] = np.asarray(
                valid[t, :y1 - gy0, :x1 - gx0], dtype=np.uint8)

    def read_coords(self, coords, di):
        n = len(coords)
        out = np.zeros((n, 768), np.float32)
        have = np.zeros(n, bool)
        if not self.enabled or self.feat is None or di is None:
            return out, have
        for k, (gy, gx) in enumerate(coords):
            if 0 <= gy < self.gh and 0 <= gx < self.gw and self.valid[gy, gx, di]:
                out[k] = self.feat[gy, gx, di, :]
                have[k] = True
        return out, have

    def write_coords(self, coords, di, feats):
        if not self.enabled or self.feat is None or di is None:
            return
        feats = np.asarray(feats, dtype=self.dtype)
        for k, (gy, gx) in enumerate(coords):
            if 0 <= gy < self.gh and 0 <= gx < self.gw:
                self.feat[gy, gx, di, :] = feats[k]
                self.valid[gy, gx, di] = 1

    def flush(self):
        for arr in (self.feat, self.valid):
            if arr is not None:
                try:
                    arr.flush()
                except Exception:
                    pass

    def close(self):
        self.flush()
        self.feat = None
        self.valid = None

    def info(self):
        return {"enabled": bool(self.enabled), "dir": self.dir if self.enabled else None,
                "shape": [self.gh, self.gw, self.ndates, 768] if self.enabled else None,
                "dtype": self.dtype_name if self.enabled else None,
                "dates": self.labels if self.enabled else []}


def _feature_store(out_dir, mode, s, labels, band_idx, enabled=None, model_paths=None,
                   engine=None):
    model_sig = []
    for p in (model_paths or []):
        if p and os.path.exists(p):
            model_sig.append([os.path.abspath(p), os.path.getmtime(p)])
    return FeatureStore(out_dir, labels, (s["c0"], s["r0"], s["w"], s["h"]),
                        _source_sig(mode, s, labels), band_idx, mode,
                        FEATURE_STORE if enabled is None else enabled,
                        model_sig=model_sig, engine=engine)


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
        self.sess = ort.InferenceSession(buf, _ort_options(),
                                         providers=["CPUExecutionProvider"])

    @property
    def provider(self):
        return self.sess.get_providers()[0]

    def run(self, feats, ctx, valid, dt, doy):
        """feats (M,T,768), ctx (M,T,3,3,768), valid (M,T,3,3) -> (M,2)."""
        return self.sess.run(None, {"feats": np.ascontiguousarray(feats, np.float32),
                                    "ctx": np.ascontiguousarray(ctx, np.float32),
                                    "valid": np.asarray(valid, dtype=bool),
                                    "dt": np.ascontiguousarray(dt, np.float32),
                                    "doy": np.ascontiguousarray(doy, np.float32)})[0]


class OpenVINOEncoder:
    """Encoder on the OpenVINO CPU plugin (f32 default, bf16 optional)."""

    def __init__(self, path, precision=None):
        import openvino as ov
        self.precision = (precision or OV_PRECISION or "f32").lower()
        self.core = ov.Core()
        model = self.core.read_model(path)
        model.reshape({"vol": [OV_BATCH, 6, CHIP, CHIP]})
        cfg = {"INFERENCE_NUM_THREADS": THREADS, "NUM_STREAMS": 1,
               "PERFORMANCE_HINT": "THROUGHPUT"}
        cfg["INFERENCE_PRECISION_HINT"] = self.precision
        self.cm = self.core.compile_model(model, "CPU", cfg)
        self._provider = f"OpenVINO-CPU-{self.precision}"
        try:
            self._actual = str(self.cm.get_property("INFERENCE_PRECISION_HINT"))
        except Exception:
            self._actual = "?"

    @property
    def provider(self):
        return self._provider

    def run(self, chips):
        chips = np.ascontiguousarray(chips, dtype=np.float32)
        n = len(chips)
        if n == 0:
            return np.empty((0, 768), np.float32)
        out = []
        for i0 in range(0, n, OV_BATCH):
            chunk = chips[i0:i0 + OV_BATCH]
            k = len(chunk)
            if k < OV_BATCH:
                pad = np.zeros((OV_BATCH - k, 6, CHIP, CHIP), np.float32)
                chunk = np.concatenate([chunk, pad], 0)
            y = np.asarray(list(self.cm({"vol": chunk}).values())[0],
                           dtype=np.float32)
            out.append(y[:k])
        return out[0] if len(out) == 1 else np.concatenate(out, 0)


class EncoderModel:
    """Channel-agnostic CNN encoder (12 layers x 768, ONNX): chips -> per-date features."""

    def __init__(self, path, provider="cpu"):
        self._ov = None
        self.sess = None
        orig_path = path
        pref = str(provider or "auto").lower()
        ov_precision = None
        ov_path = path
        if pref in ("openvino-bf16", "ov-bf16"):
            ov_precision = "bf16"
            pref = "openvino"
        if pref in ("openvino", "ov", "cpu-openvino"):
            try:
                self._ov = OpenVINOEncoder(ov_path, precision=ov_precision)
                return
            except Exception as exc:
                print(f"[forest_change] OpenVINO недоступен ({exc}); ONNX Runtime",
                      flush=True)
        import onnxruntime as ort
        with open(orig_path, "rb") as fh:
            buf = fh.read()
        try:
            self.sess = ort.InferenceSession(buf, _ort_options(),
                                             providers=pick_providers(provider))
        except Exception:
            self.sess = ort.InferenceSession(buf, _ort_options(),
                                             providers=["CPUExecutionProvider"])

    @property
    def provider(self):
        if self._ov is not None:
            return self._ov.provider
        return self.sess.get_providers()[0]

    def run(self, chips):
        """chips (M,6,32,32) raw reflectance -> (M,768)."""
        if self._ov is not None:
            return self._ov.run(chips)
        chips = np.ascontiguousarray(chips, dtype=np.float32)
        if len(chips) == 0:
            return np.empty((0, 768), np.float32)
        out = []
        for i0 in range(0, len(chips), RUN_BATCH):
            out.append(self.sess.run(None, {"vol": chips[i0:i0 + RUN_BATCH]})[0])
        return out[0] if len(out) == 1 else np.concatenate(out, 0)


class TypeCfCModel:
    """TypeNet CfC with dynamic T (ONNX Loop): feats + dt + doy -> (M,3) logits.

    Runs on CPU: the ONNX Loop operator produces wrong results on DirectML.
    """

    def __init__(self, path, provider="cpu"):
        import onnxruntime as ort
        with open(path, "rb") as fh:
            buf = fh.read()
        self.sess = ort.InferenceSession(buf, _ort_options(),
                                         providers=["CPUExecutionProvider"])
        self.has_ctx = any(i.name == "ctx" for i in self.sess.get_inputs())
        self.has_tex = any(i.name == "tex" for i in self.sess.get_inputs())
        self.has_spec = any(i.name == "spec" for i in self.sess.get_inputs())

    @property
    def provider(self):
        return self.sess.get_providers()[0]

    def run(self, feats, ctx, valid, dt, doy, tex, spec):
        feed = {"feats": np.ascontiguousarray(feats, np.float32),
                "dt": np.ascontiguousarray(dt, np.float32),
                "doy": np.ascontiguousarray(doy, np.float32)}
        if self.has_ctx:
            feed["ctx"] = np.ascontiguousarray(ctx, np.float32)
            feed["valid"] = np.asarray(valid, dtype=bool)
        if self.has_tex:
            feed["tex"] = np.ascontiguousarray(tex, np.float32)
        if self.has_spec:
            feed["spec"] = np.ascontiguousarray(spec, np.float32)
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
    raw_files = list(files or []) or ([t1_path, t2_path]
                                     if t1_path and t2_path else [])
    if mode != "period":
        raw_files, aoi_bbox = _cog_inputs(raw_files, aoi_bbox, out_dir)
    s = _setup(mode, period, mosaic_dir, raw_files, None, None, aoi_bbox)
    if mode != "period":
        s["series_files"] = list(raw_files)
    labels, dt_all, kind, paths = _series_for(s, mode, dates)
    labels_all = list(labels)
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
    store = _feature_store(out_dir, mode, s, labels_all, band_idx,
                           model_paths=[enc_path],
                           engine=f"{provider}:{OV_PRECISION}")
    date_indices = [store.date_index(l) for l in labels]
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
                store.close()
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
            for i0 in range(0, n, MASK_BATCH):
                logits = model.run(feats[i0:i0 + MASK_BATCH],
                                   ctx[i0:i0 + MASK_BATCH],
                                   cval[i0:i0 + MASK_BATCH],
                                   dtb[i0:i0 + MASK_BATCH],
                                   doyb[i0:i0 + MASK_BATCH])
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
            for i0 in range(0, nr * nc, RUN_BATCH):
                f[i0:i0 + RUN_BATCH] = enc.run(chips[i0:i0 + RUN_BATCH])
            grid[li] = f.reshape(nr, nc, 768)
        first_i = series[0][:, oy:oy + th, ox:ox + tw]
        last_i = series[-1][:, oy:oy + th, ox:ox + tw]
        invalid |= (first_i[3] == 0) | (last_i[3] == 0)
        if FOREST_NDVI_MIN > 0:
            invalid |= (_ndvi(first_i) < FOREST_NDVI_MIN) & (_ndvi(last_i) < FOREST_NDVI_MIN)
        cells_changed.clear()
        nh, nw = th // CHIP, tw // CHIP
        if store.enabled and nh and nw:
            gy0, gx0 = y // CHIP, x // CHIP
            store.write_tile(
                grid[:, oy // CHIP:oy // CHIP + nh, ox // CHIP:ox // CHIP + nw, :],
                gvalid[:, oy // CHIP:oy // CHIP + nh, ox // CHIP:ox // CHIP + nw],
                gx0, gy0, date_indices)
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
               "changed_ha": float(changed_chips * CHIP * CHIP * px_area), "mask_tif": path,
               "feature_store": store.info()}
    store.close()
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


class OpenVINOCfcUNet:
    def __init__(self, path):
        import openvino as ov
        self.core = ov.Core()
        model = self.core.read_model(path)
        self.cm = self.core.compile_model(
            model, "CPU", {"INFERENCE_NUM_THREADS": THREADS,
                           "NUM_STREAMS": 1,
                           "INFERENCE_PRECISION_HINT": OV_PRECISION})
        self._provider = f"OpenVINO-CPU-{OV_PRECISION}"

    @property
    def provider(self):
        return self._provider

    def run(self, x, dt, lengths):
        out = self.cm({"x": np.ascontiguousarray(x, np.float32),
                       "dt": np.ascontiguousarray(dt, np.float32),
                       "lengths": np.ascontiguousarray(lengths, np.int64)})
        return np.asarray(list(out.values())[0], dtype=np.float32)


class CfcUNetModel:
    """CfC U-Net pixel mask: (M,T,9,32,32) -> (M,32,32) logits."""

    def __init__(self, path, provider="auto"):
        self._ov = None
        self.sess = None
        if str(provider or "auto").lower() in ("openvino", "ov", "cpu-openvino",
                                               "openvino-bf16"):
            try:
                self._ov = OpenVINOCfcUNet(path)
                return
            except Exception as exc:
                print(f"[forest_change] U-Net OpenVINO недоступен ({exc}); ORT",
                      flush=True)
        import onnxruntime as ort
        with open(path, "rb") as fh:
            buf = fh.read()
        self.sess = ort.InferenceSession(buf, _ort_options(),
                                         providers=["CPUExecutionProvider"])

    @property
    def provider(self):
        if self._ov is not None:
            return self._ov.provider
        return self.sess.get_providers()[0]

    def run(self, x, dt, lengths):
        if self._ov is not None:
            return self._ov.run(x, dt, lengths)
        return self.sess.run(None, {"x": np.ascontiguousarray(x, np.float32),
                                    "dt": np.ascontiguousarray(dt, np.float32),
                                    "lengths": np.ascontiguousarray(lengths, np.int64)})[0]


def _remove_small(mask, min_px):
    """Exact 8-connected component filter via run-length union-find."""
    mask = np.asarray(mask, bool)
    if min_px <= 1 or not mask.any():
        return mask
    h, w = mask.shape
    parent, size = [], []

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        a, b = find(a), find(b)
        if a == b:
            return
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]

    rows = []
    prev = []
    for y in range(h):
        row = mask[y]
        cur = []
        x = 0
        while x < w:
            if not row[x]:
                x += 1
                continue
            x0 = x
            while x < w and row[x]:
                x += 1
            lab = len(parent)
            parent.append(lab)
            size.append(x - x0)
            cur.append((x0, x - 1, lab))
        for x0, x1, lab in cur:
            for px0, px1, plab in prev:
                if x0 <= px1 + 1 and px0 <= x1 + 1:
                    union(lab, plab)
        rows.append(cur)
        prev = cur
    out = np.zeros((h, w), bool)
    for y, cur in enumerate(rows):
        for x0, x1, lab in cur:
            if size[find(lab)] >= min_px:
                out[y, x0:x1 + 1] = True
    return out


def compute_disturbance(mode, out_dir, period=None, mosaic_dir=None, files=None, t1_path=None,
                        t2_path=None, band_idx=None, aoi_bbox=None, progress=None,
                        mask_path=None, preset="standard", unet_model=None,
                        provider="auto"):
    """Pixel-level disturbance mask: CfC U-Net inside the model-changed chips.

    The U-Net consumes all selected dates (T, 9 features: 6 bands + NDVI/NBR/NDMI)
    for every 32 px chip, threshold 0.438 + component filter min_px=11 (EA).
    Falls back to dNDVI/dNBR if the ONNX is missing.
    """
    _BOA_MODE.clear()
    _BOA_AMBIG.clear()
    os.makedirs(out_dir, exist_ok=True)
    raw_files = list(files or []) or ([t1_path, t2_path]
                                     if t1_path and t2_path else [])
    if mode != "period":
        raw_files, aoi_bbox = _cog_inputs(raw_files, aoi_bbox, out_dir)
    s = _setup(mode, period, mosaic_dir, raw_files, None, None, aoi_bbox)
    if mode != "period":
        s["series_files"] = list(raw_files)
    labels, dt_all, _kind, _paths = _series_for(s, mode, None)
    if len(labels) < 2:
        raise ValueError("Для пиксельной маски нужно минимум 2 даты")
    mask_file = mask_path or os.path.join(out_dir, "mask.tif")
    if not os.path.exists(mask_file):
        raise ValueError(f"Маска не найдена: {mask_file}")
    unet_path = unet_model or os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "cfc_unet_p3.onnx")
    use_unet = os.path.exists(unet_path)
    unet = CfcUNetModel(unet_path, provider) if use_unet else None
    T = len(labels)
    dt_sel = np.zeros(T, np.float32)
    for k in range(1, T):
        dt_sel[k] = float(dt_all[k])
    mgt, _, _, _ = _grid_from(mask_file)
    mask_dx = int(round((s["gt"][0] + s["c0"] * s["gt"][1] - mgt[0]) / s["gt"][1]))
    mask_dy = int(round((s["gt"][3] + s["r0"] * s["gt"][5] - mgt[3]) / s["gt"][5]))
    path = os.path.join(out_dir, "disturbance.tif")
    ds = _create_raster(path, s["gt"], s["c0"], s["r0"], s["w"], s["h"], s["proj"])
    band = ds.GetRasterBand(1)
    tiles = _tiles(s["w"], s["h"])
    dist_px = chip_px = 0
    for ti, (x, y, tw, th) in enumerate(tiles):
        m = _read_band1(mask_file, mask_dx + x, mask_dy + y, tw, th)
        changed = m == 1
        out = np.full((th, tw), 255, np.uint8)
        chip_px += int(changed.sum())
        if not changed.any():
            band.WriteArray(out, x, y)
            if progress:
                progress(int(100 * (ti + 1) / len(tiles)))
            continue
        reads = []
        for label in labels:
            b, _scl = _series_read(mode, s, band_idx, label,
                                   s["c0"] + x, s["r0"] + y, tw, th)
            reads.append(b)
        if use_unet:
            nh, nw = th // CHIP, tw // CHIP
            if nh and nw:
                sel = np.argwhere(changed[:nh * CHIP, :nw * CHIP].reshape(
                    nh, CHIP, nw, CHIP).any(axis=(1, 3)))
                if len(sel):
                    feats9 = [_features9(reads[t]) for t in range(T)]
                    for b0 in range(0, len(sel), CFC_BATCH):
                        b1 = min(len(sel), b0 + CFC_BATCH)
                        part = sel[b0:b1]
                        m = len(part)
                        xb = np.empty((m, T, 9, CHIP, CHIP), np.float32)
                        for t in range(T):
                            for k, (i, j) in enumerate(part):
                                xb[k, t] = feats9[t][
                                    :, i * CHIP:(i + 1) * CHIP,
                                    j * CHIP:(j + 1) * CHIP]
                        p = unet.run(xb, np.tile(dt_sel, (m, 1)),
                                     np.full(m, T, np.int64))
                        p = 1.0 / (1.0 + np.exp(-np.asarray(p, np.float32)))
                        for k, (i, j) in enumerate(part):
                            pred = _remove_small(p[k] > UNET_THR, UNET_MIN_PX)
                            rr, cc = i * CHIP, j * CHIP
                            out[rr:rr + CHIP, cc:cc + CHIP] = pred.astype(np.uint8)
                            dist_px += int(pred.sum())
        else:
            b1 = reads[0].astype(np.float32) / SCALE
            b2 = reads[-1].astype(np.float32) / SCALE
            nd1 = (b1[3] - b1[2]) / np.maximum(b1[3] + b1[2], 1e-6)
            nd2 = (b2[3] - b2[2]) / np.maximum(b2[3] + b2[2], 1e-6)
            nb1 = (b1[3] - b1[5]) / np.maximum(b1[3] + b1[5], 1e-6)
            nb2 = (b2[3] - b2[5]) / np.maximum(b2[3] + b2[5], 1e-6)
            dist = changed & (((nd2 - nd1) < -0.10) | ((nb1 - nb2) > 0.05))
            out[changed] = 0
            out[dist] = 1
            dist_px += int(dist.sum())
        out[~changed] = 255
        band.WriteArray(out, x, y)
        if progress:
            progress(int(100 * (ti + 1) / len(tiles)))
    band = None
    ds = None
    summary = {"disturbance_tif": path, "method": "cfc_unet_ea" if use_unet else "spectral",
               "preset": preset,
               "unet_thr": float(UNET_THR) if use_unet else None,
               "unet_min_px": int(UNET_MIN_PX) if use_unet else None,
               "unet_provider": unet.provider if unet is not None else None,
               "dates": labels,
               "disturbance_px": dist_px, "disturbance_ha": dist_px * 0.01,
               "chip_ha": chip_px * 0.01,
               "ratio_to_chips": dist_px / max(chip_px, 1)}
    with open(os.path.join(out_dir, "disturbance_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"[forest_change] площадь: {summary['disturbance_ha']:.1f} га "
          f"({summary['method']}) внутри {summary['chip_ha']:.1f} га чипов", flush=True)
    return summary


def predict(mode, model_path, out_dir, period=None, mosaic_dir=None, files=None, t1_path=None,
            t2_path=None, band_idx=None, aoi_bbox=None, progress=None, mask_path=None,
            provider="auto", use_intermediate=True):
    """Type classification of the changed chips (tiles without changes are skipped)."""
    _BOA_MODE.clear()
    _BOA_AMBIG.clear()
    os.makedirs(out_dir, exist_ok=True)
    raw_files = list(files or []) or ([t1_path, t2_path]
                                     if t1_path and t2_path else [])
    if mode != "period":
        raw_files, aoi_bbox = _cog_inputs(raw_files, aoi_bbox, out_dir)
    s = _setup(mode, period, mosaic_dir, raw_files, None, None, aoi_bbox)
    if mode != "period":
        s["series_files"] = list(raw_files)
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
    labels_all = list(labels)
    warn = _series_warning(labels, dt_all, mode)
    sel = _select_dates(len(labels), keep_intermediate=use_intermediate)
    labels = [labels[i] for i in sel]
    dt_sel = np.zeros(len(sel), np.float32)
    for k in range(1, len(sel)):
        dt_sel[k] = float(dt_all[sel[k - 1] + 1:sel[k] + 1].sum())
    doy_sel = np.stack([_doy3(l, year=(kind != "mosaic")) for l in labels])
    enc = EncoderModel(enc_path, provider)
    cfc = TypeCfCModel(model_path, provider)
    store = _feature_store(out_dir, mode, s, labels_all, band_idx,
                           model_paths=[enc_path],
                           engine=f"{provider}:{OV_PRECISION}")
    date_indices = [store.date_index(l) for l in labels]
    print(f"[forest_change] С‚РёРїС‹: TypeNet (СЌРЅРєРѕРґРµСЂ {os.path.basename(enc_path)} + "
          f"cross-attention readout {os.path.basename(model_path)}, РґРёРЅР°РјРёС‡РµСЃРєРѕРµ T, "
          f"readout РЅР° CPU{', texture 9' if cfc.has_spec else ''}), {len(labels)} РґР°С‚, "
          f"feature store {'on' if store.enabled else 'off'}", flush=True)
    cells = []
    tiles = _tiles(w, h)
    if progress:
        progress(1)

    def process_window(ax, ay, series, change):
        """Classify changed chips; reuse/encode each needed chip feature once."""
        D = len(series)
        th_sub, tw_sub = change.shape
        nh, nw = th_sub // CHIP, tw_sub // CHIP
        if nh == 0 or nw == 0:
            return
        frac = change[:nh * CHIP, :nw * CHIP].reshape(
            nh, CHIP, nw, CHIP).mean(axis=(1, 3))
        sel = np.argwhere(frac >= CHANGE_FRAC)
        if len(sel) == 0:
            return
        needed = set()
        for i, j in sel:
            i, j = int(i), int(j)
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    needed.add((min(max(i + dj, 0), nh - 1),
                                min(max(j + di, 0), nw - 1)))
        coords = sorted(needed)
        cindex = {c: k for k, c in enumerate(coords)}
        gx0, gy0 = ax // CHIP, ay // CHIP
        store_coords = [(gy0 + i, gx0 + j) for i, j in coords]
        K = len(coords)
        feat = np.empty((K, D, 768), np.float32)
        have = np.zeros((K, D), bool)
        for d, di in enumerate(date_indices):
            if store.enabled and di is not None:
                vals, avail = store.read_coords(store_coords, di)
                feat[avail, d] = vals[avail]
                have[avail, d] = True
        for d, di in enumerate(date_indices):
            miss = np.where(~have[:, d])[0]
            if len(miss) == 0:
                continue
            raw = np.empty((len(miss), 6, CHIP, CHIP), np.float32)
            for q, k in enumerate(miss):
                i, j = coords[k]
                raw[q] = series[d][:, i * CHIP:(i + 1) * CHIP,
                                   j * CHIP:(j + 1) * CHIP].astype(np.float32) / SCALE
            vals = enc.run(raw)
            feat[miss, d] = vals
            have[miss, d] = True
            if store.enabled and di is not None:
                store.write_coords([store_coords[k] for k in miss], di, vals)
        M = len(sel)
        feats = np.empty((M, D, 768), np.float32)
        ctx = np.empty((M, D, 3, 3, 768), np.float32)
        tex = np.empty((M, D, 9), np.float32)
        spec = np.empty((M, D, 9), np.float32)
        for n, (i, j) in enumerate(sel):
            i, j = int(i), int(j)
            for d in range(D):
                feats[n, d] = feat[cindex[(i, j)], d]
                raw = series[d][:, i * CHIP:(i + 1) * CHIP,
                                j * CHIP:(j + 1) * CHIP].astype(np.float32)
                ref = raw / SCALE
                tex[n, d] = _texture9(np.clip(ref, 0.0, 1.5))
                spec[n, d] = _features9(raw).mean(axis=(1, 2))
                for dj in (-1, 0, 1):
                    for di in (-1, 0, 1):
                        ii = min(max(i + dj, 0), nh - 1)
                        jj = min(max(j + di, 0), nw - 1)
                        ctx[n, d, dj + 1, di + 1] = feat[cindex[(ii, jj)], d]
        for b0 in range(0, M, CFC_BATCH):
            b1 = min(M, b0 + CFC_BATCH)
            m = b1 - b0
            logits = cfc.run(feats[b0:b1], ctx[b0:b1],
                             np.ones((m, D, 3, 3), np.float32),
                             np.tile(dt_sel, (m, 1)),
                             np.tile(doy_sel, (m, 1, 1)),
                             tex[b0:b1], spec[b0:b1])
            logits = np.asarray(logits, dtype=np.float32) + TYPE_BIAS
            probs = _softmax(logits)
            for q in range(m):
                k = int(logits[q].argmax()) + 1
                i, j = sel[b0 + q]
                rr = ay + int(i) * CHIP
                cc = ax + int(j) * CHIP
                cells.append((rr, cc, k, float(probs[q, k - 1])))

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
        # write the classes of the cells found in this tile
        for (rr, cc, k, _conf) in cells[n_before:]:
            out[rr - y:rr - y + CHIP, cc - x:cc - x + CHIP] = k
        out[invalid_tile if use_mask else invalid] = 255
        band.WriteArray(out, x, y)
        if progress:
            progress(int(100 * (ti + 1) / len(tiles)))
    ex.shutdown()
    store.close()
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
               "feature_store": store.info(),
               "classes_tif": cls_path, "changes_gpkg": gpkg, "cells_gpkg": cells_path}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    return summary


def _create_raster(path, gt, c0, r0, w, h, proj, nodata=255):
    _DS_CACHE.pop(path, None)
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
    txn = False
    try:
        layer.StartTransaction()
        txn = True
    except Exception:
        pass
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
    if txn:
        layer.CommitTransaction()
    dst = None


def _polygonize(classes_path, gpkg_path, proj):
    src = gdal.Open(classes_path)
    W, H = src.RasterXSize, src.RasterYSize
    gt = src.GetGeoTransform()
    tmp = classes_path + ".mask.tif"
    if os.path.exists(tmp):
        os.remove(tmp)
    mds = gdal.GetDriverByName("GTiff").Create(
        tmp, W, H, 1, gdal.GDT_Byte, options=["TILED=YES", "COMPRESS=DEFLATE"])
    mds.SetGeoTransform(gt)
    mds.SetProjection(proj)
    mb = mds.GetRasterBand(1)
    for x, y, tw, th in _tiles(W, H):
        arr = src.GetRasterBand(1).ReadAsArray(x, y, tw, th)
        if arr is not None:
            mb.WriteArray(((arr > 0) & (arr < 200)).astype(np.uint8), x, y)
    mds = None
    mask_ds = gdal.Open(tmp)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    if os.path.exists(gpkg_path):
        os.remove(gpkg_path)
    dst = ogr.GetDriverByName("GPKG").CreateDataSource(gpkg_path)
    layer = dst.CreateLayer("changes", srs, ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("cls", ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn("type", ogr.OFTString))
    layer.CreateField(ogr.FieldDefn("area_ha", ogr.OFTReal))
    gdal.Polygonize(src.GetRasterBand(1), mask_ds.GetRasterBand(1), layer, 0,
                    [], callback=None)
    txn = False
    try:
        layer.StartTransaction()
        txn = True
    except Exception:
        pass
    try:
        for feat in layer:
            geom = feat.GetGeometryRef()
            if geom is None:
                continue
            c = feat.GetField("cls")
            area = round(geom.GetArea() / 1e4, 3)
            if MIN_AREA_HA > 0 and area < MIN_AREA_HA:
                layer.DeleteFeature(feat.GetFID())
                continue
            feat.SetField("type", CLASSES.get(c, str(c)))
            feat.SetField("area_ha", area)
            layer.SetFeature(feat)
        if txn:
            layer.CommitTransaction()
    except Exception:
        if txn:
            layer.RollbackTransaction()
        raise
    finally:
        dst = None
        mask_ds = None
        src = None
        if os.path.exists(tmp):
            os.remove(tmp)
