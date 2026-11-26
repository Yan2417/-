import csv
import json
import math
from pathlib import Path
import numpy as np
from pyproj import Transformer
import shapely
from shapely.geometry import shape,mapping
from shapely.ops import unary_union
from scipy.optimize import milp, Bounds, LinearConstraint

root=Path.cwd()
baseline=root.parent/"exposure"
units={f["properties"]["unit"]:f for f in json.loads((baseline/"exposure_baseline.geojson").read_text())["features"]}
flood={f["properties"]["unit"]:shape(f["geometry"]) for f in json.loads((root/"gain_polygons.geojson").read_text())["features"]}
osm=json.loads((baseline/"osm_jan01_clipped.geojson").read_text(encoding="utf-8"))["features"]
roads=[(shape(f["geometry"]),f["properties"]) for f in osm if f["properties"]["layer"]=="roads"]
buildings=unary_union([shape(f["geometry"]) for f in osm if f["properties"]["layer"]=="buildings_a"])
to_m=Transformer.from_crs(4326,32649,always_xy=True).transform
def metric(g):
    return shapely.transform(g,to_m,interleaved=False)
def write_table(name,records):
    with (root/name).open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
th=np.array(json.loads((root.parent/"assessment/intensity.json").read_text())["fraction_thresholds"])
def grade(p):return 1+int(np.searchsorted(th,p,side="right"))
with (root/"q3_units.csv").open() as f:q3=list(csv.DictReader(f))
rows=[]
for cell in q3:
    i=cell["unit"]; geom=flood[i]; u=units[i]["properties"]
    intersections=[(metric(line.intersection(geom)).length/1000,attrs) for line,attrs in roads if line.intersects(geom)]
    overlap=sum(length for length,_ in intersections)
    classified=sum(length for length,attrs in intersections if attrs["fclass"] in {"motorway","motorway_link","trunk","trunk_link","primary","primary_link","secondary","secondary_link"})
    bridge_ids={attrs["osm_id"] for length,attrs in intersections if length>0 and attrs.get("bridge")=="T"}
    pop=json.loads((root/"worldpop_queries"/(i+".json")).read_text())
    # The polygon query may round individual fragment areas. Compare with our original pixel-area calculation.
    assert abs(pop["area_km2"]-float(cell["gain"])) < .03
    assert pop["total_population"] <= u["population_estimate"]+1
    lower,point,upper=map(float,[cell["new_lower_frac"],cell["new_obs_frac"],cell["new_upper_frac"]])
    rows.append(dict(unit=i,new_km2=float(cell["gain"]),coverage=1-float(cell["unknown"])/float(cell["unit_km2"]),
                     fraction=point,grade=grade(point),grade_low=grade(lower),grade_high=grade(upper),
                     population=u["population_estimate"],exposed=pop["total_population"],
                     road_overlap_km=overlap,mapped_road_km=u["mapped_road_km"],
                     secondary_or_higher_overlap_km=classified,intersected_bridge_records=len(bridge_ids),
                     building_overlap_m2=metric(geom.intersection(buildings)).area,
                     uniform_exposed=u["population_estimate"]*lower))
def normalized(key):
    a=np.array([r[key] for r in rows]); return a/max(a.max(),1e-12)
haz=.8*normalized("fraction")+.2*(np.array([r["grade"] for r in rows])-1)/3
people=normalized("exposed");access=normalized("road_overlap_km")
scores=lambda w:100*(w[0]*haz+w[1]*people+w[2]*access)
rank=lambda s:np.argsort(np.argsort(-s,kind="stable"),kind="stable")+1
score=scores([.45,.40,.15]);order=rank(score)
variants=[[.45,.40,.15],[.55,.30,.15],[.30,.55,.15],[.35,.35,.30]]
ranks=np.stack([rank(scores(w)) for w in variants])
for k,r in enumerate(rows):
    r.update(score=score[k],rank=int(order[k]),rank_best=int(ranks[:,k].min()),rank_worst=int(ranks[:,k].max()))
write_table("q4_risk.csv",sorted(rows,key=lambda r:r["rank"]))
out=[]
for r in rows:
    out.append(dict(type="Feature",geometry=units[r["unit"]]["geometry"],properties=r))
(root/"q4_risk.geojson").write_text(json.dumps(dict(type="FeatureCollection",features=out)))

def solve(demand,stock,fair=.45):
    minimum=np.ceil(fair*demand).astype(int)
    if minimum.sum()>stock:raise ValueError("Stock cannot meet the coverage floor")
    available=min(int(demand.sum()),stock)
    # Each undelivered kit incurs a unit penalty between one and two.
    weight=1+score/100
    result=milp(-weight,integrality=np.ones(len(rows)),
                bounds=Bounds(minimum,demand),
                constraints=LinearConstraint(np.ones((1,len(rows))),available,available))
    assert result.success
    x=np.rint(result.x).astype(int)
    assert np.all(x>=minimum) and np.all(x<=demand) and x.sum()==available
    # Independent exchange-optimality check: no kit can be moved to a higher-priority unmet unit.
    assert not any(weight[i]>weight[j]+1e-9 and x[i]<demand[i] and x[j]>minimum[j]
                   for i in range(len(x)) for j in range(len(x)))
    return x
def proportional(demand,stock):
    quotas=min(stock,int(demand.sum()))*demand/demand.sum()
    x=np.floor(quotas).astype(int)
    rem=round(quotas.sum()-x.sum())
    x[np.argsort(-(quotas-x),kind="stable")[:rem]]+=1
    return x
d=np.ceil([r["exposed"] for r in rows]).astype(int)
def outcome(x,d):
    return dict(delivered=int(x.sum()),unmet=int((d-x).sum()),minimum_coverage=float((x/d).min()),
                weighted_shortage=float(((d-x)*(1+score/100)).sum()))
stock=3000
x=solve(d,stock); comp=proportional(d,stock)
allocation=[dict(unit=r["unit"],priority=r["rank"],estimated_people=r["exposed"],demand=int(d[k]),
                 kits=int(x[k]),remaining=int(d[k]-x[k]),coverage=x[k]/d[k],proportional=int(comp[k]))
            for k,r in enumerate(rows)]
write_table("q5_allocation.csv",sorted(allocation,key=lambda r:r["priority"]))
sensitivity=[]
for k in [2400,3000,3600]:
    for mult in [.8,1.,1.2]:
        dem=np.ceil(np.array([r["exposed"] for r in rows])*mult).astype(int)
        if np.ceil(.45*dem).sum()>k:
            sensitivity.append(dict(stock=k,demand_multiplier=mult,feasible=False,total_demand=int(dem.sum())))
        else:
            xx=solve(dem,k)
            sensitivity.append(dict(stock=k,demand_multiplier=mult,feasible=True,total_demand=int(dem.sum()),**outcome(xx,dem)))
summary=dict(matched_units=len(rows),exposed=sum(r["exposed"] for r in rows),resident_population=sum(r["population"] for r in rows),
             road_overlap_km=sum(r["road_overlap_km"] for r in rows),mapped_road_km=sum(r["mapped_road_km"] for r in rows),
             building_overlap_m2=sum(r["building_overlap_m2"] for r in rows),uniform_population_alternative=sum(r["uniform_exposed"] for r in rows),
             fixed_grade_units=sum(r["grade_low"]==r["grade_high"] for r in rows),
             total_demand=int(d.sum()),stock=stock,optimized=outcome(x,d),proportional=outcome(comp,d),sensitivity=sensitivity)
(root/"q45_results.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
