"""Recheck saved rasters and recompute evaluation independently of training."""
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import baseline as b

def load_csv(file):
    with open(file) as f: return list(csv.DictReader(f))

def read(file):
    with Image.open(file) as im:
        assert im.size==(256,256)
        geo=tuple(tuple(im.tag_v2[k]) for k in [33550,33922,34735])
        return np.array(im),geo

def main():
    split=load_csv("split_manifest.csv"); scored=load_csv("test_tiles.csv")
    metrics=json.loads(Path("metrics.json").read_text())
    assert len(split)==1000 and len({r["id"] for r in split})==1000
    tests={r["id"] for r in split if r["split"]=="test"}
    assert tests=={r["id"] for r in scored}=={f.stem for f in Path("test_predictions").glob("*.tif")}
    for r in split:
        i=r["id"]; d=Path("registered")/i
        headers=[]
        for name in ["pre_rgb","post_sar","label","valid"]:
            with Image.open(d/(name+".tif")) as im:
                assert im.size==(256,256)
                headers.append(tuple(tuple(im.tag_v2[k]) for k in [33550,33922,34735]))
        assert all(h==headers[0] for h in headers)
    sums=np.zeros((2,2),np.float64)
    for r in scored:
        i=r["id"]; d=Path("registered")/i
        v,vg=read(d/"valid.tif"); y,yg=read(d/"label.tif"); pred,pg=read(Path("test_predictions")/(i+".tif"))
        assert vg==yg==pg
        assert np.array_equal(pred!=255,v!=0)
        valid=v.astype(bool); assert valid.sum()==int(r["valid_pixels"])
        cm=np.bincount(2*y[valid]+pred[valid],minlength=4)
        assert np.array_equal(cm,[int(r[k]) for k in ["tn","fp","fn","tp"]])
        # Independent midpoint ellipsoid integration over each pixel row (km2).
        (dx,dy,_),origin,_=pg
        phi=np.radians(origin[4]-(np.arange(256)+.5)*dy)
        ecc2=0.0066943799901413165
        weight=(6378.137**2*(1-ecc2)*np.cos(phi)*math.radians(dx)*math.radians(dy)
                /(1-ecc2*np.sin(phi)**2)**2)
        for ref in (0,1):
            for est in (0,1):
                area=np.dot(weight,np.sum((y==ref)&(pred==est)&valid,axis=1))
                sums[ref,est]+=area
                key=(("tn","fp"),("fn","tp"))[ref][est]
                assert math.isclose(area,float(r[key+"_km2"]),rel_tol=1e-8,abs_tol=2e-9)
    tn,fp,fn,tp=sums.ravel()
    m=metrics["metrics"]["all"]
    assert abs(tp/(tp+fp+fn)-m["iou"])<1e-9 and abs((fp-fn)-m["error_km2"])<1e-6
    ordered=sorted((r for r in scored if r["iou"]),key=lambda r:float(r["iou"]))
    examples=[ordered[round((len(ordered)-1)*q)] for q in (.25,.5,.75)]
    forest=cv2.ml.RTrees_load("forest.xml.gz")
    for r in examples:
        i=int(r["id"]); _,ok,votes=b.votes(forest,{"id":i})
        actual,_=read(Path("test_predictions")/f"{i}.tif")
        assert np.array_equal(np.where(ok,votes>=metrics["threshold_votes"],255).astype("uint8"),actual)
    result=dict(registered_grids_checked=1000,predictions_checked=len(tests),area_crosscheck="WGS84 midpoint quadrature",
                reloaded_model_bit_exact_ids=[r["id"] for r in examples],verified_iou=float(tp/(tp+fp+fn)),
                illustrative_tiles=[{k:r[k] for k in ("id","region","iou","true_km2","pred_km2")} for r in examples])
    b.dump("verification.json",result)
    print(json.dumps(result))
if __name__=="__main__": main()
