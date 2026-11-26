"""Sample official ESA WorldCover v200 (2021) COG blocks at test pixel centres."""
import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import math
import os
import time
import urllib.request
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

ROOT="https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
TILE_CELLS=36000


def fetch(url,first,last):
    for attempt in range(4):
        try:
            request=urllib.request.Request(url,headers={"Range":f"bytes={first}-{last}"})
            with urllib.request.urlopen(request,timeout=40) as response:
                if response.status!=206:
                    raise RuntimeError(f"Expected a partial response for {url}, got {response.status}")
                data=response.read()
                if len(data)!=last-first+1:
                    raise ValueError("Truncated byte range")
                return data
        except (OSError,ValueError) as err:
            if attempt==3: raise
            time.sleep(2**attempt)


def run(base,out):
    out.mkdir(parents=True,exist_ok=True)
    cache=out/"blocks"; cache.mkdir(exist_ok=True)
    masks=out/"by_tile"; masks.mkdir(exist_ok=True)
    with (base/"split_manifest.csv").open() as f:
        rows=[r for r in csv.DictReader(f) if r["split"]=="test"]
    original_limit=Image.MAX_IMAGE_PIXELS
    for long0 in (111,114):
        tag=f"N27E{long0}"
        name=f"ESA_WorldCover_10m_2021_v200_{tag}_Map.tif"
        url=ROOT+name
        headfile=cache/(tag+".header")
        if not headfile.exists(): headfile.write_bytes(fetch(url,0,131071))
        try:
            # This is a known 36000x36000 COG. Only parse tags, never decode the full scene.
            Image.MAX_IMAGE_PIXELS=None
            with Image.open(io.BytesIO(headfile.read_bytes())) as im:
                t=im.tag_v2
                assert im.size==(TILE_CELLS,TILE_CELLS) and t[259]==8 and t[317]==1 and t[258]==(8,)
                dx,dy=t[33550][:2]; west,north=t[33922][3:5]; bw,bh=t[322],t[323]
                offsets=t[324]; lengths=t[325]
        finally:
            Image.MAX_IMAGE_PIXELS=original_limit
        nx=math.ceil(TILE_CELLS/bw)
        pixels={}
        needed=set()
        for r in rows:
            if not long0<=float(r["lon"])<long0+3:continue
            cols=np.floor((float(r["x"])+(np.arange(256)+.5)*float(r["dx"])-west)/dx).astype(int)
            rr=np.floor((north-float(r["y"])+(np.arange(256)+.5)*float(r["dy"]))/dy).astype(int)
            assert 0<=rr.min()<=rr.max()<TILE_CELLS and 0<=cols.min()<=cols.max()<TILE_CELLS
            pixels[r["id"]]=(rr,cols)
            needed.update(int(i*nx+j) for i in np.unique(rr//bh) for j in np.unique(cols//bw))
        def read_block(k):
            save=cache/f"{tag}.{k}.deflate"
            if not save.exists():
                raw=fetch(url,offsets[k],offsets[k]+lengths[k]-1)
                save.write_bytes(raw)
            raw=save.read_bytes()
            array=np.frombuffer(zlib.decompress(raw),np.uint8).reshape(bh,bw)
            return k,array,hashlib.sha256(raw).hexdigest()
        blocks={}; checks={}
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            for k,array,sha in pool.map(read_block,sorted(needed)):
                blocks[k]=array
                checks[k]=sha
        for i,(rr,cols) in pixels.items():
            values=np.zeros((256,256),np.uint8)
            for y in np.unique(rr//bh):
                for x in np.unique(cols//bw):
                    iy=np.flatnonzero(rr//bh==y); ix=np.flatnonzero(cols//bw==x)
                    values[np.ix_(iy,ix)]=blocks[int(y*nx+x)][np.ix_(rr[iy]%bh,cols[ix]%bw)]
            Image.fromarray(values).save(masks/(i+".png"))
        (out/(tag+"_sources.json")).write_text(json.dumps(dict(url=url,product="ESA WorldCover 2021 v200",native_grid=[west,north,dx,dy],
          block_sha256=checks,header_sha256=hashlib.sha256(headfile.read_bytes()).hexdigest()),indent=2))
        print(tag,len(needed),"blocks",len(pixels),"registered tiles",flush=True)
    assert set(p.stem for p in masks.glob("*.png"))=={r["id"] for r in rows}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--baseline",required=True,type=Path);p.add_argument("--out",required=True,type=Path)
    a=p.parse_args();run(a.baseline,a.out)
