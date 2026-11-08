"""Out-of-process runner: keeps QGIS alive on OOM and streams progress as JSON."""
import contextlib
import json
import os
import sys
import traceback

from . import core


def _progress(p):
    print(json.dumps({"progress": int(p)}), file=sys.__stdout__, flush=True)


def _run(payload):
    stage = payload["stage"]
    out_dir = payload["out_dir"]
    mode = "period" if payload.get("period") else "files"
    base = dict(period=payload.get("period"),
                mosaic_dir=payload.get("mosaic_dir"),
                files=payload.get("files"),
                band_idx=tuple(payload["band_idx"]) if payload.get("band_idx") else None,
                aoi_bbox=tuple(payload["aoi_bbox"]) if payload.get("aoi_bbox") else None)
    provider = payload.get("provider", "auto")
    model_dir = os.path.join(os.path.dirname(os.path.abspath(core.__file__)), "model")
    if stage in ("mask", "all"):
        mask = core.compute_mask(
            mode, out_dir,
            mask_model=os.path.join(model_dir, "masknet_cfc.onnx"),
            use_intermediate=bool(payload.get("mask_intermediate",
                                             payload.get("use_intermediate", True))),
            force_mask=bool(payload.get("force_mask", False)),
            provider=provider, progress=_progress, **base)
    else:
        mask = None
    if stage in ("types", "all"):
        types = core.predict(
            mode, os.path.join(model_dir, "typenet_cfc.onnx"), out_dir,
            use_intermediate=bool(payload.get("types_intermediate",
                                             payload.get("use_intermediate", True))),
            mask_path=os.path.join(out_dir, "mask.tif"),
            provider=provider, progress=_progress, **base)
    else:
        types = None
    if stage == "area":
        return core.compute_disturbance(
            mode, out_dir, mask_path=os.path.join(out_dir, "mask.tif"),
            preset=payload.get("preset", "standard"),
            unet_model=os.path.join(model_dir, "cfc_unet_p3.onnx"),
            provider=provider, progress=_progress, **base)
    if stage == "all":
        return {"mask": mask, "types": types}
    return mask if stage == "mask" else types


def main(argv=None):
    argv = argv or sys.argv
    try:
        payload = json.loads(argv[1]) if len(argv) > 1 else json.load(sys.stdin)
        with contextlib.redirect_stdout(sys.stderr):
            result = _run(payload)
        print(json.dumps({"result": result}, ensure_ascii=False, default=str),
              file=sys.__stdout__, flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc), "traceback": traceback.format_exc()},
                         ensure_ascii=False), file=sys.__stdout__, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
