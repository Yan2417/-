"""Transfer the frozen Q2 network to the 7 July operator quicklook, at its native scale."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from pyproj import Transformer
import rasterio as rio
from rasterio.warp import reproject,Resampling
from rasterio.transform import from_bounds
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from event_map import ROOT, HENG, BBOX
from predict_scene import prepare,load_model

def fit_registration():
    src=np.array(Image.open(ROOT/"sources/operator_0707.png").convert("RGB"))
    ref=np.array(Image.open(ROOT/"sources/registration_reference.jpg").convert("RGB"))
    colour=cv2.erode((np.ptp(src,2)>8).astype("uint8")*255,np.ones((5,5),np.uint8))
    sift=cv2.SIFT_create(12000,contrastThreshold=.012)
    a,da=sift.detectAndCompute(cv2.cvtColor(src,cv2.COLOR_RGB2GRAY),colour)
    b,db=sift.detectAndCompute(cv2.cvtColor(cv2.resize(ref,None,fx=.8,fy=.8),cv2.COLOR_RGB2GRAY),None)
    matches=[p for p,q in cv2.BFMatcher().knnMatch(da,db,k=2) if p.distance<.8*q.distance]
    x=np.float32([a[m.queryIdx].pt for m in matches]); y=np.float32([b[m.trainIdx].pt for m in matches])/.8
    hold=np.arange(len(x))%4==0
    cv2.setRNGSeed(2417)
    M,ok=cv2.estimateAffinePartial2D(x[~hold],y[~hold],method=cv2.RANSAC,ransacReprojThreshold=2)
    err=np.linalg.norm(x[hold]@M[:,:2].T+M[:,2]-y[hold],axis=1)
    ext=json.loads((ROOT/"sources/reference_extent.json").read_text(encoding="utf-8-sig"))
    e=ext["extent"]; g=from_bounds(e["xmin"],e["ymin"],e["xmax"],e["ymax"],ext["width"],ext["height"])
    # SIFT indices locate centres, while the raster affine uses pixel corners.
    geo=g*rio.Affine.translation(.5,.5)*rio.Affine(*M.ravel())*rio.Affine.translation(-.5,-.5)
    stats=dict(matches=len(matches),fit_inliers=int(ok.sum()),heldout=len(err),heldout_within_2px=int((err<2).sum()),
               holdout_median_reference_px=float(np.median(err)),holdout_p90_reference_px=float(np.quantile(err,.90)),
               native_ground_metres_approx=abs(geo.a)*np.cos(np.radians(22.9)))
    assert stats["holdout_median_reference_px"]<1 and ok.sum()>80
    return src,geo,stats

def rgb_on_preview(geo,shape):
    with rio.open(HENG/"pre_rgb.tif") as f:
        # Pool clear observations only. Retain dates and masks in the original optical mosaic.
        valid=(f.dataset_mask()>0).astype("float32")
        rgb=f.read()
        def avg(a):
            out=np.zeros(shape,np.float32)
            reproject(a,out,src_transform=f.transform,src_crs=f.crs,dst_transform=geo,dst_crs="EPSG:3857",
                      resampling=Resampling.average)
            return out
        coverage=avg(valid)
        result=np.stack([avg(b.astype("float32")*valid)/np.maximum(coverage,1e-7) for b in rgb],axis=2)
        return np.clip(np.rint(result),0,255).astype("uint8"),coverage

def infer(model,rgb,sar,valid):
    # Same patch size as training, but each display pixel covers about 96 m.
    out=np.full(sar.shape,np.nan,np.float32)
    rr,cc=np.where(valid)
    r0,c0=rr.min(),cc.min(); r1,c1=rr.max()+1,cc.max()+1
    with torch.inference_mode():
        for y in range(r0,r1,256):
            for x in range(c0,c1,256):
                sl=np.s_[y:min(y+256,r1),x:min(x+256,c1)]
                mask=valid[sl]
                if not mask.any():continue
                h,w=mask.shape
                tensor=prepare(rgb[sl],sar[sl],mask)
                scores=model(tensor[None]).sigmoid()[0,:h,:w].numpy()
                out[sl]=np.where(mask,scores,np.nan)
    return out

def reference_july7(lon,lat):
    # One graticule tick per axis in Fig 5. Find scale independently by matching
    # the drawn river against Fig 3 with its two known longitude ticks.
    fig3=np.array(Image.open(ROOT/"sources/Shan2026_Fig3.png").convert("RGB")).astype(int)
    fig5=np.array(Image.open(ROOT/"sources/Shan2026_Fig5.png").convert("RGB")).astype(int)
    def blue(im):
        r,g,b=im.transpose(2,0,1)
        return (b-r>95)&(b-g>60)&(b>160)&(g>r+12)
    ys,xs=np.where(blue(fig3))
    window=(ys>590)&(ys<850)&(xs>960)&(xs<1600)
    qlon=109.25+(xs[window]-425.5)/10596
    qlat=22+55/60-(ys[window]-663.5)/10596
    dist=cv2.distanceTransform((~blue(fig5)).astype("uint8"),cv2.DIST_L2,3)
    trials=[]
    for scale in range(16000,33001,25):
        x=np.rint(1556+(qlon-(109+20/60))*scale).astype(int)
        y=np.rint(564+((22+55/60)-qlat)*scale).astype(int)
        trials.append((np.minimum(dist[y.clip(0,2065),x.clip(0,2549)],16).mean(),scale))
    _,scale=min(trials)
    x=np.rint(1556+(lon-(109+20/60))*scale).astype(int)
    y=np.rint(564+((22+55/60)-lat)*scale).astype(int)
    ok=(x>101)&(x<2490)&(y>135)&(y<1680)
    v=fig5[y.clip(0,2065),x.clip(0,2549)]
    r,g,b=v.transpose(2,0,1)
    water=(r-b>55)&(r-g>10)&(r>165)&(g>53)
    ok&=~((b-r>48)|(g-r>15))
    # Mask the overprinted village names and highway band conservatively.
    for left,top,right,bottom in [(36,450,225,489),(904,492,1090,542),(820,192,999,239),(249,246,464,291)]:
        ok&=~((x>=left*2550/1100)&(x<=right*2550/1100)&(y>=top*2066/891)&(y<=bottom*2066/891))
    return water,ok,scale

def registered_radar(refit):
    """Cache the numeric swath and its valid mask, keeping the three-pixel texture halo."""
    folder = ROOT / "sources"
    raster = folder / "July07_registered_swath.tif"
    metadata = folder / "July07_registration.json"
    if refit or not raster.exists() or not metadata.exists():
        src, geo, stats = fit_registration()
        swath = np.zeros(src.shape[:2], np.uint8)
        cv2.fillConvexPoly(swath, np.int32([[156,2],[468,2],[375,501],[56,437]]), 1)
        mask = ((swath > 0) & (np.ptp(src,2) <= 5)).astype(np.uint8)
        halo = cv2.dilate(mask, np.ones((7,7), np.uint8))
        radar = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        with rio.open(raster, "w", driver="GTiff", height=swath.shape[0], width=swath.shape[1],
                      count=2, dtype="uint8", crs="EPSG:3857", transform=geo, compress="deflate") as dst:
            dst.write(np.where(halo, radar, 0).astype(np.uint8), 1)
            dst.write(mask, 2)
            dst.set_band_description(1, "display intensity inside swath and texture halo")
            dst.set_band_description(2, "valid grayscale SAR swath")
        metadata.write_text(json.dumps(stats, indent=2))
    with rio.open(raster) as ds:
        return ds.read(1), ds.read(2) == 1, ds.transform, json.loads(metadata.read_text())

def run(refit=False):
    torch.set_num_threads(2)
    sar,swath,geo,stats=registered_radar(refit)
    h,w=sar.shape
    col,row=np.meshgrid(np.arange(w)+.5,np.arange(h)+.5)
    mx,my=geo*(col,row)
    lon,lat=Transformer.from_crs(3857,4326,always_xy=True).transform(mx,my)
    roi=(lon>=BBOX[0])&(lon<BBOX[2])&(lat>=BBOX[1])&(lat<BBOX[3])
    rgb,coverage=rgb_on_preview(geo,(h,w))
    valid=swath&(coverage>=.75)
    model,threshold=load_model(ROOT.parent/"q2/fusion_best.pt")
    reference,ref_valid,scale=reference_july7(lon,lat)
    native_km2=abs(geo.a*geo.e-geo.b*geo.d)*np.cos(np.deg2rad(lat))**2/1e6
    # Web Mercator-to-ellipsoid correction at the local latitude.
    ecc=.0066943799901413165
    native_km2*= (1-ecc)/(1-ecc*np.sin(np.deg2rad(lat))**2)**2
    src_hist=np.bincount(sar[valid],minlength=256).cumsum()
    hist_path=ROOT/"sources/training_sar_histogram.json"
    if hist_path.exists():
        target=np.array(json.loads(hist_path.read_text())["counts"])
    else:
        target=np.zeros(256)
        with (ROOT.parent/"q1/split_manifest.csv").open() as f:
            for row in csv.DictReader(f):
                if row["split"]=="train":
                    reg=ROOT.parent/"q1/registered"/row["id"]
                    x=np.array(Image.open(reg/"post_sar.tif"))[::8,::8]
                    m=np.array(Image.open(reg/"valid.tif"))[::8,::8]>0
                    target+=np.bincount(x[m],minlength=256)
        hist_path.write_text(json.dumps(dict(method="every eighth row/column, valid pixels of 602 training tiles",
                                             counts=target.astype(int).tolist())))
    lut=np.searchsorted(target.cumsum()/target.sum(),src_hist/src_hist[-1]).clip(0,255).astype("uint8")
    stats.update(reference_scale_px_per_degree=scale,roi_valid_km2=float(native_km2[valid&roi].sum()),
                 source="Dianjian-1 7 July noon display raster; unknown display stretch and polarization",
                 threshold=threshold,valid_optical_fraction_min=.75)
    for key,data in [("display_dn",sar),("train_histogram",lut[sar])]:
        score=infer(model,rgb,data,valid)
        decision=score>=threshold
        m=roi & valid
        ev=m & ref_valid
        cellarea=lambda mask:float(native_km2[mask].sum())
        stats[key]=dict(predicted_km2=cellarea(m&decision),reference_comparison_km2=cellarea(ev),
                        union_reference_water_km2=cellarea(ev&reference),
                        proxy_iou=cellarea(ev&decision&reference)/max(1e-10,cellarea(ev&(decision|reference))),
                        threshold_sensitivity={str(t):cellarea(m&(score>=t)) for t in [.55,.65,.75]})
        out=np.where(m,decision,255).astype("uint8")
        with rio.open(ROOT/("July07_Q2_"+key+".tif"),"w",driver="GTiff",width=w,height=h,count=1,
                      dtype="uint8",nodata=255,crs="EPSG:3857",transform=geo,compress="deflate") as dst:
            dst.write(out,1)
    (ROOT/"q3_model_transfer.json").write_text(json.dumps(stats,indent=2))
    print(stats)
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refit-registration",action="store_true",help="recompute the transform from the original preview and optical basemap")
    run(parser.parse_args().refit_registration)
