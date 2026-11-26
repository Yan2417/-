"""Reconstruct the visible flood class in Shan et al. (2026), Fig. 3, inside 12 downstream cells."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio as rio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from PIL import Image

ROOT = Path.cwd()
HENG = ROOT.parent / "hengzhou"
BBOX = (109.22, 22.88, 109.3625, 22.92)
HEIGHT, WIDTH = 424, 1512
GRID = from_bounds(*BBOX, WIDTH, HEIGHT)

def aligned(path, invalid=255):
    dest = np.full((HEIGHT, WIDTH), invalid, np.uint8)
    with rio.open(path) as ds:
        reproject(rio.band(ds,1), dest, src_transform=ds.transform, src_crs=ds.crs,
                  dst_transform=GRID, dst_crs="EPSG:4326", dst_nodata=invalid, resampling=Resampling.nearest)
    return dest

def thematic_map(xshift=0, yshift=0, contrast=76):
    colour = np.asarray(Image.open(ROOT/"sources/Shan2026_Fig3.png").convert("RGB"))
    assert colour.shape[:2] == (1458,1800)
    px = 425.5 + ((GRID.c + (np.arange(WIDTH)+.5)*GRID.a) - 109.25)*10596 + xshift
    py = 663.5 + (22+55/60 - (GRID.f+(np.arange(HEIGHT)+.5)*GRID.e))*10596 + yshift
    cols, rows = np.meshgrid(np.rint(px).astype(int),np.rint(py).astype(int))
    r,g,b = colour[rows,cols].astype(int).transpose(2,0,1)
    red = (r-g > contrast) & (r-b > contrast-15)
    # Blue pre-event polygons can hide the after-event overlay. River/village
    # lines, roads and text are cartographic symbols, not observed land cover.
    ambiguous_blue = (b-r>60) & (g-r>52)
    blue_line = (b-r>55) & (b-g>30)
    mint = (g-r>18) & (b-r>5) & (g>216)
    yellow_line = (r-b>30) & (g-b>10) & (r>160) & ~red
    def box(x0,y0,x1,y1):
        return (cols>=x0)&(rows>=y0)&(cols<=x1)&(rows<=y1)
    labels = (box(802,765,1147,831)|box(520,990,836,1045)|box(1440,607,1734,658)
              |box(1490,941,1780,1000)|box(855,1081,973,1134)|box(74,962,207,1028))
    ink = labels & (np.maximum.reduce([r,g,b]) < 95)
    # Road outlines/antialias fringe adjacent to the coloured cartographic lines.
    obscured = ambiguous_blue | (cv2.dilate((blue_line|mint|yellow_line|ink).astype("uint8"),
                                              np.ones((3,3),np.uint8))>0)
    return red, ~obscured

def row_km2():
    lat = GRID.f + (np.arange(HEIGHT)+.5)*GRID.e
    e2 = 1-(1-1/298.257223563)**2
    return (6378.137**2*(1-e2)*np.cos(np.deg2rad(lat))*np.deg2rad(GRID.a)*np.deg2rad(-GRID.e)
            /(1-e2*np.sin(np.deg2rad(lat))**2)**2)[:,None]

def write_tif(name, array):
    with rio.open(ROOT/name,"w",driver="GTiff",height=HEIGHT,width=WIDTH,count=1,dtype="uint8",
                  crs="EPSG:4326",transform=GRID,nodata=255,compress="deflate") as ds:
        ds.write(array.astype("uint8"),1)

def main():
    pre = aligned(HENG/"pre_water_Jul04_sar.tif")
    optical = aligned(HENG/"pre_optical_water.tif")
    post, readable = thematic_map()
    valid = readable & (pre!=255)
    change = np.where(valid,2*(pre==1)+post,255).astype("uint8")
    write_tif("July06_transitions.tif",change)
    write_tif("July04_radar_water_candidate.tif",pre)
    write_tif("July06_visible_water.tif",np.where(readable,post,255))
    weights = np.broadcast_to(row_km2(),(HEIGHT,WIDTH))
    classes = dict(dry=0,gain=1,loss=2,persistent=3,unknown=255)
    features, rows = [], []
    for i in range(2):
        for j in range(6):
            sl = np.s_[i*212:(i+1)*212,j*252:(j+1)*252]
            part, w = change[sl],weights[sl]
            areas = {k:float(w[part==v].sum()) for k,v in classes.items()}
            area = float(w.sum())
            new = areas["gain"]
            row = dict(unit=f"r{i+5}c{j+1}",unit_km2=area,**areas,new_lower_frac=new/area,
                       new_upper_frac=(new+areas["unknown"])/area,
                       new_obs_frac=new/(area-areas["unknown"]))
            rows.append(row)
            tf=GRID*rio.Affine.translation(j*252,i*212)
            polys=[shape(g) for g,_ in shapes((part==1).astype("uint8"),mask=part==1,transform=tf)]
            original=unary_union(polys)
            simplified=original.simplify(.000075,preserve_topology=True)
            features.append(dict(type="Feature",properties={"unit":row["unit"],"simplified_area_ratio":simplified.area/original.area if polys else None},
                                 geometry=mapping(simplified)))
    (ROOT/"gain_polygons.geojson").write_text(json.dumps(dict(type="FeatureCollection",features=features)))
    total = {k:sum(r[k] for r in rows) for k in ["unit_km2",*classes]}
    total.update(pre_common=total["loss"]+total["persistent"],post_common=total["gain"]+total["persistent"],
                 net_common=total["gain"]-total["loss"],valid_km2=total["unit_km2"]-total["unknown"])
    total["fraction_common"]=total["gain"]/total["valid_km2"]
    total["total_fraction_lower"]=total["gain"]/total["unit_km2"]
    total["total_fraction_upper"]=(total["gain"]+total["unknown"])/total["unit_km2"]
    assert np.isclose(sum(total[k] for k in classes),total["unit_km2"])
    assert np.isclose(total["post_common"]-total["pre_common"],total["net_common"])
    sens=[]
    for dx,dy,t in [(0,0,76),(2,0,76),(-2,0,76),(0,2,76),(0,-2,76),(0,0,56),(0,0,96)]:
        red,ok=thematic_map(dx,dy,t)
        sens.append(dict(dx_map_px=dx,dy_map_px=dy,contrast=t,gain_km2=float(weights[ok&(pre==0)&red].sum())))
    opt_ok=(optical!=255)&valid
    total["optical_comparison"]=dict(common_km2=float(weights[opt_ok].sum()),
                        gain_using_sar_pre=float(weights[opt_ok & post & (pre==0)].sum()),
                        gain_using_optical_pre=float(weights[opt_ok & post & (optical==0)].sum()),
                        pre_water_sar=float(weights[opt_ok & (pre==1)].sum()),
                        pre_water_optical=float(weights[opt_ok & (optical==1)].sum()))
    (ROOT/"q3_results.json").write_text(json.dumps(dict(bbox=BBOX,metrics=total,sensitivity=sens),indent=2))
    with (ROOT/"q3_units.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=rows[0]);wr.writeheader();wr.writerows(rows)
    print(total)

if __name__=="__main__":
    main()
