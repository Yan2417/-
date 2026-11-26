"""Reopen saved Q2 predictions, recompute areas independently and test checkpoint reload."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from fusion import FusionNet, Tiles, get_array, manifest, probability_batches
from baseline import dump


def verify(base, result, out):
    rows=manifest(base); tests=[r for r in rows if r["split"]=="test"]
    with (result/"test_tiles.csv").open() as f:
        by_id={int(r["id"]):r for r in csv.DictReader(f)}
    assert set(by_id)=={r["id"] for r in tests}=={int(p.stem) for p in (result/"test_predictions").glob("*.tif")}
    total=np.zeros(4)
    for r in tests:
        i=str(r["id"]); root=base/"registered"/i
        y=get_array(root/"label.tif"); ok=get_array(root/"valid.tif")==1
        with Image.open(result/"test_predictions"/(i+".tif")) as im, Image.open(root/"label.tif") as lab:
            assert im.size==lab.size and all(im.tag_v2[k]==lab.tag_v2[k] for k in (33550,33922,34735))
            pred=np.array(im)
        assert np.array_equal(pred!=255,ok)
        lat=np.radians(r["y"]-(np.arange(256)+.5)*r["dy"]); e2=.0066943799901413165
        rw=(6378.137**2*(1-e2)*np.cos(lat)*np.radians(r["dx"])*np.radians(r["dy"])/(1-e2*np.sin(lat)**2)**2)
        w=np.broadcast_to(rw[:,None],ok.shape)
        area=np.bincount(2*y[ok]+pred[ok],weights=w[ok],minlength=4)
        np.testing.assert_allclose(area,[float(by_id[int(i)][k+"_km2"]) for k in ("tn","fp","fn","tp")],atol=3e-9,rtol=1e-8)
        total+=area
    tn,fp,fn,tp=total; iou=float(tp/(tp+fp+fn))
    ref=json.loads((result/"fusion_metrics.json").read_text())["test"]["all"]
    assert abs(iou-ref["iou"])<1e-9 and abs((fp-fn)-ref["error_km2"])<1e-6
    # Repeat the inference batch layout used in evaluation, avoiding mixed-precision batch-shape effects.
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state=torch.load(result/"fusion_best.pt",map_location=device)
    model=FusionNet().to(device); model.load_state_dict(state["state"])
    data=Tiles(base,tests[:12])
    repeated=[]
    for probs,ys,oks,indices in probability_batches(model,data,12,device):
        for p,ok,k in zip(probs,oks,indices):
            i=tests[k]["id"]
            actual=get_array(result/"test_predictions"/f"{i}.tif")
            assert np.array_equal(actual,np.where(ok,p>=state["threshold"],255).astype("uint8"))
            repeated.append(i)
    dump(out,dict(test_masks=len(tests),checked_registered_grids=len(tests),pooled_iou=iou,
                  area_error_km2=float(fp-fn),checkpoint_reproduced_ids=repeated))
    print("Verified",len(tests),"test masks and",len(repeated),"reloaded predictions.")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--baseline",required=True,type=Path);p.add_argument("--fusion",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args();verify(a.baseline,a.fusion,a.out)
