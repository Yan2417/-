"""Train the Q2 optical/SAR fusion network; freeze model/threshold before test evaluation."""
import argparse
import csv
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

import baseline as b

SEED = 2417
THRESHOLDS = np.arange(30, 76, 5) / 100.0
EPOCHS = 24


def manifest(base):
    with (base / "split_manifest.csv").open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("id","valid_pixels"):
            r[k] = int(r[k])
        for k in ("x","y","dx","dy","lon","lat","west","east","north","south"):
            r[k] = float(r[k])
    return rows


def val_parts(rows):
    parts = {}
    for region in ("west","east"):
        rr = [r for r in rows if r["split"]=="validation" and r["region"]==region]
        center = np.median([r["lat"] for r in rr])
        for r in rr:
            if r["north"] < center-.0046:
                role = "select"
            elif r["south"] > center+.0046:
                role = "calibrate"
            else:
                role = "val_buffer"
            parts[r["id"]] = role
    return parts


def get_array(p):
    with Image.open(p) as im:
        return np.array(im)


def load(base, r):
    path = base / "registered" / str(r["id"])
    rgb = get_array(path/"pre_rgb.tif")
    radar = get_array(path/"post_sar.tif")
    valid = get_array(path/"valid.tif")
    y = (get_array(path/"label.tif")==1).astype(np.uint8)
    # Median and local standard deviation preserve SAR texture alongside raw intensity.
    x = radar.astype(np.float32)
    mean = cv2.blur(x,(7,7))
    sd = np.sqrt(np.maximum(cv2.blur(x*x,(7,7))-mean*mean,0))
    inputs = np.dstack((rgb,radar,cv2.medianBlur(radar,3),np.clip(sd,0,255).astype("uint8")))
    inputs[valid==0]=0
    return np.ascontiguousarray(inputs.transpose(2,0,1)),y,valid


class Tiles(torch.utils.data.Dataset):
    def __init__(self, base, rows, augment=False):
        self.rows=rows
        self.arrays=[load(base,r) for r in rows]
        self.augment=augment

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        x,y,m=self.arrays[idx]
        if self.augment:
            k=random.randrange(4)
            x=np.rot90(x,k,axes=(1,2)); y=np.rot90(y,k); m=np.rot90(m,k)
            if random.random()<.5:
                x=x[:,:,::-1]; y=y[:,::-1]; m=m[:,::-1]
        return (torch.from_numpy(np.ascontiguousarray(x)),torch.from_numpy(np.ascontiguousarray(y)),
                torch.from_numpy(np.ascontiguousarray(m)),idx)


def block(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.SiLU(),
                         nn.Conv2d(b,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.SiLU())


class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.opt=block(3,8)
        self.sar=block(3,8)
        self.enc=nn.ModuleList([block(16,24),block(24,40),block(40,64)])
        self.deep=block(64,96)
        self.dec=nn.ModuleList([block(96+64,64),block(64+40,40),block(40+24,24),block(24+16,16)])
        self.head=nn.Conv2d(16,1,1)

    def forward(self,x):
        x=x.float()/127.5-1
        z=torch.cat((self.opt(x[:,:3]),self.sar(x[:,3:])),1)
        skips=[z]
        for enc in self.enc:
            z=enc(F.max_pool2d(z,2)); skips.append(z)
        z=self.deep(F.max_pool2d(z,2))
        for dec,skip in zip(self.dec,reversed(skips)):
            z=dec(torch.cat((F.interpolate(z,scale_factor=2,mode="bilinear",align_corners=False),skip),1))
        return self.head(z)[:,0]


def loss_function(logits,y,m):
    y=y.float(); m=m.float(); dims=(1,2)
    p=logits.float().sigmoid()
    n=m.sum(dim=dims).clamp_min(1)
    bce=(F.binary_cross_entropy_with_logits(logits.float(),y,reduction="none")*m).sum()/n.sum()
    dice=1-(2*(p*y*m).sum()+1)/((p*m).sum()+(y*m).sum()+1)
    area=(((p-y)*m).sum(dim=dims)/n).abs().mean()
    return bce+.65*dice+1.8*area


@torch.no_grad()
def probability_batches(model, dataset, batch_size, device):
    model.eval()
    for x,y,m,idx in torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False):
        with torch.cuda.amp.autocast(enabled=device.type=="cuda"):
            scores=model(x.to(device)).float().sigmoid()
        yield scores.cpu().numpy(),y.numpy(),m.numpy().astype(bool),idx.numpy()


def selection(model,data,bs,device):
    # Quantize only for the histogram; threshold comparisons reproduce the 0.05 grid.
    mats=[]
    for scores,ys,oks,ids in probability_batches(model,data,bs,device):
        for p,y,ok,idx in zip(scores,ys,oks,ids):
            w=np.broadcast_to(b.pixel_areas(data.rows[idx]),y.shape)
            q=np.searchsorted(THRESHOLDS,p,side="right")
            h=np.array([np.bincount(q[ok&(y==c)],weights=w[ok&(y==c)],minlength=len(THRESHOLDS)+1) for c in (0,1)])
            mats.append(h)
    a=np.array(mats); ref=a[:,1,:].sum(1); trials=[]
    for j,t in enumerate(THRESHOLDS):
        tp=a[:,1,j+1:].sum(1); fp=a[:,0,j+1:].sum(1)
        iou=tp.sum()/(ref.sum()+fp.sum())
        wape=np.abs(tp+fp-ref).sum()/ref.sum()
        trials.append(dict(threshold=float(t),iou=float(iou),wape=float(wape),objective=float(wape+1-iou)))
    return min(trials,key=lambda v:(v["objective"],abs(v["threshold"]-.5)))


def score_tile(r,y,ok,pred):
    a=b.pixel_areas(r)[:,0]
    row={k:r[k] for k in ["id","region","lon","lat","valid_pixels"]}
    for name,k in (("tp",pred&(y==1)),("fp",pred&(y==0)),("fn",~pred&(y==1)),("tn",~pred&(y==0))):
        px=(k&ok).sum(1)
        row[name]=int(px.sum()); row[name+"_km2"]=float(px@a)
    tp,fp,fn=[row[k+"_km2"] for k in ("tp","fp","fn")]
    row.update(full_km2=float(a.sum()*256),valid_km2=float(ok.sum(1)@a),
               true_km2=tp+fn,pred_km2=tp+fp,signed_error_km2=fp-fn,abs_error_km2=abs(fp-fn),
               abs_relative_error=b.ratio(abs(fp-fn),tp+fn),iou=b.ratio(tp,tp+fp+fn),dice=b.ratio(2*tp,2*tp+fp+fn))
    return row


def evaluate(model,base,rows,t,group,bs,device):
    data=Tiles(base,rows); records=[]
    dst=Path(group+"_predictions"); dst.mkdir(exist_ok=True)
    for scores,ys,oks,indices in probability_batches(model,data,bs,device):
        for p,y,ok,i in zip(scores,ys,oks,indices):
            r=rows[i]; pred=p>=t
            row=score_tile(r,y,ok,pred); records.append(row)
            b.write_geo(dst/f"{r['id']}.tif",np.where(ok,pred,255).astype("uint8"),[r[k] for k in ("x","y","dx","dy")],255)
    b.table(group+"_tiles.csv",records)
    return {k:b.summary([r for r in records if k=="all" or r["region"]==k]) for k in ("all","west","east")}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--batch-size",type=int,default=12)
    args=parser.parse_args(); base=args.baseline.resolve(); bs=args.batch_size
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows=manifest(base); parts=val_parts(rows)
    b.table("validation_roles.csv",[dict(id=k,role=v) for k,v in parts.items()])
    train=Tiles(base,[r for r in rows if r["split"]=="train"],augment=True)
    select=Tiles(base,[r for r in rows if parts.get(r["id"])=="select"])
    model=FusionNet().to(device)
    loader=torch.utils.data.DataLoader(train,batch_size=bs,shuffle=True,num_workers=0,
                                       generator=torch.Generator().manual_seed(SEED))
    opt=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.0005)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=.00012)
    scaler=torch.cuda.amp.GradScaler(enabled=device.type=="cuda")
    best=float("inf"); trace=[]
    print("train",len(train),"select",len(select),"GPU",device,"epochs",EPOCHS,flush=True)
    for epoch in range(1,EPOCHS+1):
        model.train(); epoch_loss=0.
        for x,y,m,_ in loader:
            opt.zero_grad(set_to_none=True)
            x=x.to(device); y=y.to(device); m=m.to(device)
            with torch.cuda.amp.autocast(enabled=device.type=="cuda"):
                z=model(x); loss=loss_function(z,y,m)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
            scaler.step(opt); scaler.update()
            epoch_loss+=loss.item()*len(x)
        schedule.step()
        trial=selection(model,select,bs,device)
        row=dict(epoch=epoch,loss=epoch_loss/len(train),**trial); trace.append(row)
        if trial["objective"]<best:
            best=trial["objective"]
            torch.save(dict(state=model.state_dict(),threshold=trial["threshold"],epoch=epoch),"fusion_best.pt")
        b.table("training_curve.csv",trace)
        print(row,flush=True)
    weights=torch.load("fusion_best.pt",map_location=device)
    model.load_state_dict(weights["state"]); t=weights["threshold"]
    b.dump("frozen_model.json",dict(epoch=weights["epoch"],threshold=t,seed=SEED,objective="WAPE + 1 - pooled area IoU",
                                  select_score=best,epochs=EPOCHS))
    print("Model and threshold frozen",weights["epoch"],t,flush=True)
    out={}
    for group,rr in [("calibration",[r for r in rows if parts.get(r["id"])=="calibrate"]),
                     ("test",[r for r in rows if r["split"]=="test"])]:
        out[group]=evaluate(model,base,rr,t,group,bs,device)
    b.dump("fusion_metrics.json",out)
    print(out["test"]["all"],flush=True)

if __name__=="__main__":
    main()
