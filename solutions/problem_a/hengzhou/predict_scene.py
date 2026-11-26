"""Predict new inundation from aligned 8-bit pre-event RGB and post-event GF-3 HH.

The caller must match the training radiometry; raw DN, linear power and dB rasters
cannot be substituted without an independently documented conversion. Input masks
must encode unobserved pixels. Supply --valid for an additional 0/1 pixel mask.
"""
import argparse
import os
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
from rasterio.windows import Window

root = Path.cwd().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion import FusionNet


def prepare(rgb, radar, valid):
    h,w = radar.shape
    rgb = np.pad(rgb, ((0,256-h),(0,256-w),(0,0)), mode="edge")
    radar = np.pad(radar, ((0,256-h),(0,256-w)), mode="edge")
    x = radar.astype("float32")
    mean = cv2.blur(x,(7,7))
    sd = np.sqrt(np.maximum(cv2.blur(x*x,(7,7)) - mean*mean, 0))
    features = np.dstack((rgb, radar, cv2.medianBlur(radar,3), np.clip(sd,0,255).astype("uint8")))
    features[:h,:w][~valid] = 0
    features[h:] = 0
    features[:,w:] = 0
    return torch.from_numpy(np.ascontiguousarray(features.transpose(2,0,1)))


def load_model(checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = FusionNet()
    model.load_state_dict(state["state"])
    return model.eval(), float(state["threshold"])


def require_grid(ref, src):
    if not ref.crs or ref.crs != src.crs or ref.shape != src.shape or not ref.transform.almost_equals(src.transform, precision=1e-10):
        raise ValueError("Images must share a CRS, exact pixel grid and dimensions")


@torch.inference_mode()
def predict(model, threshold, optical, hh, out, extra_mask=None):
    from contextlib import ExitStack
    if out.resolve() in {p.resolve() for p in [optical, hh, extra_mask] if p}:
        raise ValueError("Output must be distinct from inputs")
    tmp = out.with_name(out.stem + "." + uuid.uuid4().hex + ".tmp.tif")
    try:
        with ExitStack() as stack:
            rgb = stack.enter_context(rasterio.open(optical))
            sar = stack.enter_context(rasterio.open(hh))
            mask = stack.enter_context(rasterio.open(extra_mask)) if extra_mask else None
            require_grid(rgb, sar)
            if mask:
                require_grid(rgb, mask)
            if rgb.count != 3 or sar.count != 1 or set(rgb.dtypes+sar.dtypes) != {"uint8"}:
                raise ValueError("Expected training-compatible three-band uint8 RGB and one-band uint8 HH")
            dst = stack.enter_context(rasterio.open(tmp, "w", driver="GTiff", width=rgb.width, height=rgb.height,
                        crs=rgb.crs, transform=rgb.transform, count=1, dtype="uint8", nodata=255, compress="deflate"))
            for top in range(0, rgb.height, 256):
                for left in range(0, rgb.width, 256):
                    w, h = min(256,rgb.width-left), min(256,rgb.height-top)
                    window = Window(left,top,w,h)
                    valid = np.all(rgb.read_masks(window=window) != 0, axis=0) & (sar.read_masks(1,window=window) != 0)
                    if mask:
                        valid &= (mask.read(1,window=window) == 1) & (mask.read_masks(1,window=window) != 0)
                    prediction = np.full((h,w),255,np.uint8)
                    if valid.any():
                        x = prepare(rgb.read(window=window).transpose(1,2,0), sar.read(1,window=window), valid)
                        probability = model(x[None]).sigmoid()[0,:h,:w].numpy()
                        prediction[valid] = (probability[valid] >= threshold).astype("uint8")
                    dst.write(prediction, 1, window=window)
            dst.update_tags(meaning="0 other valid; 1 newly inundated; 255 unknown", threshold=str(threshold))
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["optical","hh","output"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--valid",type=Path)
    parser.add_argument("--checkpoint",type=Path,default=root/"q2"/"fusion_best.pt")
    args=parser.parse_args()
    torch.set_num_threads(2)
    model,threshold=load_model(args.checkpoint)
    predict(model,threshold,args.optical,args.hh,args.output,args.valid)
