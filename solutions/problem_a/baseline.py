"""Run from an empty experiment directory: python -u baseline.py --source '.../Attachment 1 Model Training Dataset'."""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import PIL
from PIL import Image, TiffImagePlugin
from scipy.ndimage import distance_transform_edt

SEED = 2417
TREES = 48
PIXEL_SAMPLE_RATE = 0.008
THRESHOLDS = np.arange(12, 37, 2)  # counts of positive tree votes (of 48)
INSPECTED_LABELS = {1103, 1217, 1410, 1447, 1462, 1660, 1688, 1816, 1894}
FEATURE_NAMES = ["r", "g", "b", "g_minus_r_norm", "g_minus_b_norm", "rgb_mean7",
                 "rgb_sd7", "sar", "sar_median3", "sar_mean7", "sar_sd7", "sar_mean15"]
DEG = np.pi / 180


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def table(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def files(src, i):
    return [src / "pre-disaster optical images" / f"before_s2_{i}.TIF",
            src / "post-disaster SAR images" / f"after_gf3_hh_{i}.TIF",
            src / "label information" / f"label_{i}.tif"]


def grid(im):
    tags = im.tag_v2
    assert im.size == (256, 256) and 34264 not in tags
    anchor = tags[33922]
    assert len(anchor) == 6 and anchor[:3] == (0., 0., 0.)
    keys = dict((k, v) for k, loc, n, v in np.asarray(tags[34735][4:]).reshape(-1, 4) if loc == 0 and n == 1)
    assert keys[1024] == 2 and keys[1025] == 1 and keys[2054] == 9102
    assert keys[2048] == 4326 or (keys[2048] == 32767 and 'WGS 84' in tags[34737] and
                                   np.allclose(tags[34736][:2], [6378137, 298.257223563]))
    x, y = anchor[3:5]
    dx, dy = tags[33550][:2]
    return (float(x), float(y), float(dx), float(dy))


def write_geo(path, arr, g, nodata=None):
    x, y, dx, dy = g
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[33550] = (dx, dy, 0.)
    ifd[33922] = (0., 0., 0., x, y, 0.)
    ifd[34735] = (1, 1, 0, 4, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326, 2054, 0, 1, 9102)
    if nodata is not None:
        ifd[42113] = str(nodata)
    Image.fromarray(arr).save(path, tiffinfo=ifd, compression="tiff_lzw")


def align(arr, src, dst):
    x, y, dx, dy = dst
    ox, oy, sx, sy = src
    h, w = arr.shape[:2]
    ux = ((x-ox) + (np.arange(w)+.5)*dx) / sx - .5
    vy = ((oy-y) + (np.arange(h)+.5)*dy) / sy - .5
    mx, my = np.meshgrid(ux.astype("float32"), vy.astype("float32"))
    cover = (mx >= -.5) & (mx < w-.5) & (my >= -.5) & (my < h-.5)
    result = cv2.remap(arr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return result, cover


def prepare(src):
    found = []
    for path in files(src, 1001):
        found.append({int(p.stem.rsplit("_",1)[-1]) for p in path.parent.iterdir() if p.suffix.lower()==".tif"})
    assert found[0] == found[1] == found[2] == set(range(1001, 2001))
    records = []
    for i in sorted(found[0]):
        mats, grids = [], []
        for path in files(src, i):
            with Image.open(path) as im:
                grids.append(grid(im))
                mats.append(np.array(im))
        opt, sar, truth = mats
        assert opt.shape == (256,256,4) and sar.shape == truth.shape == (256,256)
        assert all(a.dtype == np.uint8 for a in mats) and np.isin(truth,[0,1,255]).all()
        ref = grids[2]
        rgb, ok_rgb = align(opt[:,:,:3], grids[0], ref)
        # Four-channel all-zero imagery marks blank scene edges. SAR zero can be genuine.
        support, _ = align(np.any(opt!=0,axis=2).astype(np.float32), grids[0], ref)
        radar, ok_sar = align(sar, grids[1], ref)
        valid = (support >= .99999) & ok_rgb & ok_sar & (truth != 255)
        output = Path("registered")/str(i)
        output.mkdir(parents=True)
        for name, arr, nd in [("pre_rgb",rgb,None),("post_sar",radar,None),("label",truth,255),("valid",valid.astype("uint8"),None)]:
            write_geo(output/(name+".tif"),arr,ref,nd)
        x,y,dx,dy=ref
        records.append(dict(id=i,region="west" if x<115 else "east",lon=x+128*dx,lat=y-128*dy,
                            x=x,y=y,dx=dx,dy=dy,west=min(g[0] for g in grids),east=max(g[0]+256*g[2] for g in grids),
                            south=min(g[1]-256*g[3] for g in grids),north=max(g[1] for g in grids),
                            valid_pixels=int(valid.sum())))
    return records


def separation(a, b):
    # Conservative lower bound in km across this dataset's 28-30 deg north latitude range.
    dx = max(0., a["west"]-b["east"], b["west"]-a["east"])*95.
    dy = max(0., a["south"]-b["north"], b["south"]-a["north"])*110.
    return math.hypot(dx,dy)


def split(records):
    bounds={}
    for region in ("west","east"):
        rows=[a for a in records if a["region"]==region]
        centers=np.array([a["lon"] for a in rows])
        # Place cut lines between separate tile columns, never through a column.
        cols=np.unique(centers.round(3))
        gaps=(cols[1:]+cols[:-1])[np.diff(cols)>.01]/2
        cuts=[float(gaps[np.argmin(np.abs(gaps-np.quantile(centers,q)))]) for q in (.4,.6,.85)]
        bounds[region]=cuts
        for a in rows:
            x=a["lon"]
            a["split"]="test" if cuts[0]<x<cuts[1] else ("validation" if x>cuts[2] else "train")
    held=[r for r in records if r["split"] in ("validation","test")]
    for r in records:
        if r["split"]=="train" and any(separation(r,s)<1.0 for s in held):
            r["split"]="buffer"
    minimum={}
    for sa,sb in [("train","test"),("train","validation"),("validation","test")]:
        dist=min(separation(a,b) for a in records if a["split"]==sa for b in records if b["split"]==sb)
        assert dist>=1.0
        minimum[f"{sa}_{sb}_km_lower_bound"]=dist
    assert not any(r["id"] in INSPECTED_LABELS and r["split"]=="test" for r in records)
    return dict(cuts_longitude=bounds,minimum_footprint_distances=minimum,
                counts=dict(Counter(r["split"] for r in records)),
                previous_label_inspection_test_overlap=0)


def read_tile(r):
    root=Path("registered")/str(r["id"])
    arrays=[]
    for name in ("pre_rgb","post_sar","label","valid"):
        with Image.open(root/(name+".tif")) as im:
            arrays.append(np.array(im))
    return *arrays[:3],arrays[3].astype(bool)


def features(rgb,sar,ok):
    if ok.any() and not ok.all():
        # Extend from nearest valid pixel solely as padding for neighborhood filters.
        nn=distance_transform_edt(~ok, return_distances=False, return_indices=True)
        rgb, sar = rgb[tuple(nn)], sar[tuple(nn)]
    rgb=rgb.astype("float32"); s=sar.astype("float32")
    red,green,blue=rgb.transpose((2,0,1)); brightness=rgb.mean(axis=2)
    def moments(a,k):
        mean=cv2.blur(a,(k,k))
        return mean,np.sqrt(np.maximum(cv2.blur(a*a,(k,k))-mean*mean,0))
    bm,bs=moments(brightness,7); sm,ss=moments(s,7)
    return np.stack([red,green,blue,(green-red)/(green+red+1),(green-blue)/(green+blue+1),bm,bs,s,
                     cv2.medianBlur(sar,3),sm,ss,cv2.blur(s,(15,15))],axis=2).astype("float32")


def train(rows):
    rng=np.random.default_rng(SEED)
    xs,ys=[],[]
    for r in rows:
        rgb,sar,y,ok=read_tile(r)
        idx=np.flatnonzero(ok & (rng.random(ok.shape)<PIXEL_SAMPLE_RATE))
        if len(idx):
            xs.append(features(rgb,sar,ok).reshape(-1,len(FEATURE_NAMES))[idx])
            ys.append(y.ravel()[idx])
    X,Y=np.concatenate(xs),np.concatenate(ys).astype("int32")
    cv2.setRNGSeed(SEED)
    forest=cv2.ml.RTrees_create()
    forest.setMaxDepth(16); forest.setMinSampleCount(30); forest.setActiveVarCount(4)
    forest.setCalculateVarImportance(False)
    forest.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER,TREES,0.))
    assert forest.train(X,cv2.ml.ROW_SAMPLE,Y)
    forest.save("forest.xml.gz")
    return forest, dict(n=len(Y),positive=int(Y.sum()),sampling_rate=PIXEL_SAMPLE_RATE)


def votes(forest,r):
    rgb,sar,y,ok=read_tile(r)
    pred=np.zeros(ok.shape,np.uint8)
    if ok.any():
        f=features(rgb,sar,ok)[ok]
        v=forest.getVotes(f,0)
        assert v.shape==(len(f)+1,2) and list(v[0])==[0,1]
        assert np.all(v[1:].sum(axis=1)==TREES)
        pred[ok]=v[1:,1]
    return y,ok,pred


def pixel_areas(r):
    # Exact ellipsoid surface strip per longitude interval, WGS84, in square km.
    a=6378137.; e2=1-(1-1/298.257223563)**2; e=math.sqrt(e2)
    edges=(r["y"]-np.arange(257)*r["dy"])*DEG
    u=np.sin(edges)
    strip=.5*(u/(1-e2*u*u)+np.arctanh(e*u)/e)
    return (a*a*(1-e2)*(r["dx"]*DEG)*(strip[:-1]-strip[1:])/1.e6)[:,None]


def tune(forest,rows):
    hist=np.zeros((2,TREES+1),float)
    for r in rows:
        y,ok,v=votes(forest,r)
        weights=np.broadcast_to(pixel_areas(r),y.shape)
        for c in [0,1]:
            k=ok&(y==c)
            hist[c]+=np.bincount(v[k],weights=weights[k],minlength=TREES+1)
    trial=[]
    for t in THRESHOLDS:
        tp=hist[1,t:].sum(); fp=hist[0,t:].sum(); fn=hist[1,:t].sum()
        trial.append(dict(votes=int(t),threshold=t/TREES,area_weighted_iou=tp/(tp+fp+fn),
                          signed_area_error_km2=fp-fn))
    table("validation_thresholds.csv",trial)
    return max(trial,key=lambda z:(z["area_weighted_iou"],-abs(z["threshold"]-.5)))["votes"]


def ratio(a,b):
    return float(a/b) if b else None


def score(forest,rows,threshold):
    Path("test_predictions").mkdir()
    scored=[]
    for r in rows:
        y,ok,v=votes(forest,r)
        pred=v>=threshold
        rowarea=pixel_areas(r)[:,0]
        out={k:r[k] for k in ["id","region","lon","lat","valid_pixels"]}
        for label,condition in [("tp",pred&(y==1)),("fp",pred&(y==0)),("fn",~pred&(y==1)),("tn",~pred&(y==0))]:
            px=(ok&condition).sum(axis=1)
            out[label]=int(px.sum())
            out[label+"_km2"]=float(rowarea@px)
        tp,fp,fn=[out[k+"_km2"] for k in ("tp","fp","fn")]
        out.update(true_km2=tp+fn,pred_km2=tp+fp,signed_error_km2=fp-fn,
                   abs_error_km2=abs(fp-fn),abs_relative_error=ratio(abs(fp-fn),tp+fn),
                   iou=ratio(tp,tp+fp+fn),dice=ratio(2*tp,2*tp+fp+fn))
        arr=np.where(ok,pred,255).astype(np.uint8)
        write_geo(Path("test_predictions")/f"{r['id']}.tif",arr,[r[k] for k in ("x","y","dx","dy")],255)
        scored.append(out)
    return scored


def summary(rows):
    tp,fp,fn,tn=[sum(r[k+"_km2"] for r in rows) for k in ("tp","fp","fn","tn")]
    pts,pfs,pns,nts=[sum(r[k] for r in rows) for k in ("tp","fp","fn","tn")]
    errors=np.array([r["signed_error_km2"] for r in rows if r["valid_pixels"]])
    ious=[r["iou"] for r in rows if r["iou"] is not None]
    rel=[r["abs_relative_error"] for r in rows if r["abs_relative_error"] is not None]
    return dict(tiles=len(rows),evaluated_tiles=len(errors),valid_pixels=pts+pfs+pns+nts,
                true_km2=tp+fn,pred_km2=tp+fp,error_km2=fp-fn,signed_relative_error=ratio(fp-fn,tp+fn),
                tile_mae_km2=float(np.abs(errors).mean()),tile_rmse_km2=float(np.sqrt(np.mean(errors**2))),
                wape=ratio(np.abs(errors).sum(),tp+fn),median_tile_abs_relative_error=float(np.median(rel)),
                tiles_for_relative_error=len(rel),fp_km2=fp,fn_km2=fn,
                iou=ratio(tp,tp+fp+fn),dice=ratio(2*tp,2*tp+fp+fn),precision=ratio(tp,tp+fp),
                recall=ratio(tp,tp+fn),pixel_accuracy=ratio(pts+nts,pts+pfs+pns+nts),
                pixel_iou=ratio(pts,pts+pfs+pns),tile_mean_iou=float(np.mean(ious)),tiles_for_mean_iou=len(ious))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    args=parser.parse_args()
    cv2.setNumThreads(2)
    print("Preparing 1,000 label-grid co-registered GeoTIFF groups",flush=True)
    rows=prepare(args.source)
    split_info=split(rows); table("split_manifest.csv",rows); dump("split_audit.json",split_info)
    print("Spatial split",split_info,flush=True)
    groups={k:[r for r in rows if r["split"]==k] for k in ("train","validation","test")}
    forest,train_info=train(groups["train"]); print("Training finished",train_info,flush=True)
    threshold=tune(forest,groups["validation"]); print("Frozen threshold",threshold,TREES,flush=True)
    scored=score(forest,groups["test"],threshold); table("test_tiles.csv",scored)
    metrics={g:summary([r for r in scored if g=="all" or r["region"]==g]) for g in ("all","west","east")}
    result=dict(metrics=metrics,train=train_info,threshold_votes=threshold,trees=TREES,
                feature_names=FEATURE_NAMES,seed=SEED,opencv=cv2.__version__,numpy=np.__version__,
                pillow=PIL.__version__,python=sys.version,valid_px_all=sum(a["valid_pixels"] for a in rows))
    dump("metrics.json",result)
    print(json.dumps(metrics["all"]),flush=True)

if __name__=="__main__":
    main()
