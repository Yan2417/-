"""Land-cover error decomposition, error drivers, empirical intensity grades and uncertainty."""
import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import baseline as b
from fusion import manifest, get_array

LAND={0:"unmapped",10:"trees",20:"shrubland",30:"grassland",40:"cropland",50:"built_up",
      60:"bare_sparse",70:"snow_ice",80:"permanent_water",90:"herbaceous_wetland",95:"mangroves",100:"moss_lichen"}
GRADE=["general","relatively_strong","strong","extremely_strong"]
MODELS=("baseline","fusion")
TARGET_COVERAGE=.90


def read_rows(p):
    with p.open() as f: return list(csv.DictReader(f))


def measures(cm):
    tn,fp,fn,tp=map(float,cm)
    return dict(tn_km2=tn,fp_km2=fp,fn_km2=fn,tp_km2=tp,valid_km2=tn+fp+fn+tp,true_km2=tp+fn,
                pred_km2=tp+fp,bias_km2=fp-fn,iou=b.ratio(tp,tp+fp+fn),recall=b.ratio(tp,tp+fn),precision=b.ratio(tp,tp+fp))


def add_cm(store,model,strata,groups,y,pred,ok,weights):
    for k in np.unique(groups[ok]):
        mask=ok&(groups==k)
        cm=np.bincount((2*y[mask]+pred[mask]).astype(int),weights=weights[mask],minlength=4)
        store[(model,strata,int(k))]+=cm


def diagnostics(base,fusion,wc,out,rows):
    buckets=defaultdict(lambda:np.zeros(4))
    patches=defaultdict(lambda:np.zeros(3))
    contributions=[]
    test=[r for r in rows if r["split"]=="test"]
    original={int(r["id"]):r for r in read_rows(base/"test_tiles.csv")}
    for r in test:
        i=str(r["id"]); path=base/"registered"/i
        y=get_array(path/"label.tif"); ok=get_array(path/"valid.tif")!=0
        land=get_array(wc/"by_tile"/(i+".png")); sar=get_array(path/"post_sar.tif")
        assert set(np.unique(land))<=LAND.keys()
        w=np.broadcast_to(b.pixel_areas(r),y.shape)
        cc,n=ndimage.label((y==1)&ok,structure=np.ones((3,3),bool))
        areas=np.bincount(cc.ravel(),weights=w.ravel(),minlength=n+1)
        bins=np.searchsorted([.01,.10,1.],areas,side="right")
        sr=original[r["id"]]
        hard=0 if not sr["iou"] else (1 if float(sr["iou"])<.30 else 2)
        for name,parent in zip(MODELS,[base,fusion]):
            p=get_array(parent/"test_predictions"/(i+".tif")); assert np.array_equal(p!=255,ok)
            add_cm(buckets,name,"landcover",land,y,p,ok,w)
            add_cm(buckets,name,"sar_dn_quarter",sar//64,y,p,ok,w)
            add_cm(buckets,name,"base_iou_group",np.full(y.shape,hard),y,p,ok,w)
            det=np.bincount(cc.ravel(),weights=(w*(p==1)).ravel(),minlength=n+1)
            for z in range(4):
                sel=(bins==z)&(np.arange(n+1)>0)
                patches[(name,z)]+=[sel.sum(),areas[sel].sum(),det[sel].sum()]
            gt=((y==1)&ok); est=((p==1)&ok)
            union=gt|est
            contributions.append(dict(id=r["id"],model=name,region=r["region"],baseline_iou_group=hard,
                                 ref_km2=float(w[gt].sum()),pred_km2=float(w[est].sum()),
                                 iou=b.ratio(w[gt&est].sum(),w[union].sum())))
    table=[]
    for (model,s,k),cm in sorted(buckets.items()):
        desc=LAND[k] if s=="landcover" else (["empty_union","baseline_iou_lt_0.30","baseline_iou_ge_0.30"][k] if s=="base_iou_group" else f"DN_{k*64}_{k*64+63}")
        table.append(dict(model=model,stratum=s,group=desc,**measures(cm)))
    b.table(out/"stratified_errors.csv",table)
    b.table(out/"patch_recall.csv",[dict(model=m,size_bin=["lt_0.01","0.01_to_0.10","0.10_to_1","ge_1"][z],
           components=int(n),ref_km2=ref,detected_km2=tp,recall=b.ratio(tp,ref)) for (m,z),(n,ref,tp) in sorted(patches.items())])
    b.table(out/"paired_tiles.csv",contributions)
    for name,parent,file in [(MODELS[0],base,"metrics.json"),(MODELS[1],fusion,"fusion_metrics.json")]:
        import json
        dat=json.loads((parent/file).read_text())
        ref=dat["metrics" if name=="baseline" else "test"]["all"]
        check=sum((cm for (m,s,k),cm in buckets.items() if m==name and s=="landcover"),np.zeros(4))
        assert abs(check[1]-ref["fp_km2"])<1e-6 and abs(check[2]-ref["fn_km2"])<1e-6
    lowids={k for k,v in original.items() if v["iou"] and float(v["iou"])<.30}
    summary={}
    for name in MODELS:
        group=[r for r in contributions if r["model"]==name and r["id"] in lowids]
        summary[name]=dict(n=len(group),wape=sum(abs(x["pred_km2"]-x["ref_km2"]) for x in group)/sum(x["ref_km2"] for x in group),
                           mean_tile_iou=float(np.mean([x["iou"] for x in group if x["iou"] is not None])),
                           **measures(buckets[name,"base_iou_group",1]))
    return summary


def intensity(base,fusion,out,rows):
    fractions=[]; areas=[]
    for r in rows:
        if r["split"]!="train" or r["valid_pixels"]!=65536: continue
        y=get_array(base/"registered"/str(r["id"])/"label.tif")
        w=b.pixel_areas(r)
        area=float(w.sum()*256); flooded=float(((y==1)*w).sum())
        if flooded>0:
            fractions.append(flooded/area); areas.append(area)
    thresholds=np.quantile(fractions,[.50,.75,.90])
    def grade(p):return int(np.searchsorted(thresholds,p,side="right"))
    calibration=read_rows(fusion/"calibration_tiles.csv")
    radius={}
    for reg in ("west","east"):
        errors=sorted(abs(float(z["pred_km2"])-float(z["true_km2"]))/float(z["valid_km2"]) for z in calibration
                      if z["region"]==reg and float(z["valid_km2"])>0)
        order=math.ceil((len(errors)+1)*TARGET_COVERAGE)
        radius[reg]=dict(n=len(errors),rank=order,half_width=errors[order-1] if order<=len(errors) else 1.)
    detail=[]; confusion=np.zeros((4,4),int); full_cm=np.zeros((4,4),int)
    counters=Counter()
    for r in read_rows(fusion/"test_tiles.csv"):
        reg=r["region"]; S=float(r["full_km2"]); V=float(r["valid_km2"]); c=V/S
        if V<=0:continue
        ap=float(r["pred_km2"]); at=float(r["true_km2"])
        p=ap/V; truth=at/V
        q=radius[reg]["half_width"]
        low=max(0,p-q); high=min(1,p+q)
        # Unknown pixels can be anywhere from entirely dry to entirely newly inundated.
        lo=c*low; hi=min(1.,c*high+max(0,1-c))
        gh,gt=grade(p),grade(truth); gl,gu=grade(lo),grade(hi)
        confusion[gt,gh]+=1
        is_full=int(r["valid_pixels"])==65536
        if is_full: full_cm[gt,gh]+=1
        counters["fraction_contained"]+=int(low<=truth<=high); counters["stable"]+=int(gl==gu)
        counters["full_grade_contained"]+=int(is_full and gl<=gt<=gu)
        counters["fully_observed"]+=int(is_full)
        detail.append(dict(id=int(r["id"]),region=reg,region_km2=S,valid_km2=V,coverage=c,new_observed_km2=ap,
           reference_observed_km2=at,observed_fraction=p,reference_observed_fraction=truth,
           estimated_full_km2=p*S,full_km2_lower=lo*S,full_km2_upper=hi*S,
           full_fraction_lower=lo,full_fraction_upper=hi,estimated_grade=GRADE[gh],observed_reference_grade=GRADE[gt],
           min_grade=GRADE[gl],max_grade=GRADE[gu],stable=int(gl==gu),fully_observed=int(is_full)))
    b.table(out/"intensity_by_tile.csv",detail)
    return dict(reference_train_tiles=len(fractions),fraction_thresholds=list(thresholds),quantiles=[.5,.75,.9],
                example_area_km2=5.,example_area_thresholds_km2=list(5*thresholds),calibration=radius,
                nominal_coverage=TARGET_COVERAGE,observed_fraction_coverage=counters["fraction_contained"]/len(detail),
                stable_grades=counters["stable"],n_test=len(detail),point_grade_agreement=confusion.trace()/confusion.sum(),
                confusion_observed=confusion.tolist(),fully_observed=counters["fully_observed"],
                full_tiles_point_grade_agreement=full_cm.trace()/full_cm.sum(),fully_observed_confusion=full_cm.tolist(),
                full_tiles_grade_interval_coverage=counters["full_grade_contained"]/counters["fully_observed"],
                grade_order=GRADE)


if __name__=="__main__":
    q=argparse.ArgumentParser()
    for k in ("baseline","fusion","wc","out"): q.add_argument("--"+k,required=True,type=Path)
    a=q.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    rs=manifest(a.baseline)
    b.dump(a.out/"difficult_cases.json",diagnostics(a.baseline,a.fusion,a.wc,a.out,rs))
    b.dump(a.out/"intensity.json",intensity(a.baseline,a.fusion,a.out,rs))
    print("Verified both models' error totals; graded all test tiles.")
