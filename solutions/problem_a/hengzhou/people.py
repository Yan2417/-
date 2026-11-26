"""Query modelled population in the mapped new-water polygons (2026 WorldPop)."""
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path

root=Path.cwd()
cache=root/"worldpop_queries"
cache.mkdir(exist_ok=True)

def query(feature):
    unit=feature["properties"]["unit"]
    target=cache/(unit+".json")
    geom=feature["geometry"]
    import hashlib
    body=json.dumps(dict(year=2026,geojson=geom),separators=(",",":")).encode()
    key=hashlib.sha256(body).hexdigest()
    if target.exists() and json.loads(target.read_text()).get("query_sha256")==key:
        return
    for attempt in range(3):
        try:
            req=urllib.request.Request("https://api.worldpop.org/v3/population",data=body,
                                        headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=55) as response:
                result=json.load(response)
            assert result["data_year"]==2026 and result["total_population"]>=0
            result.update(query_sha256=key,unit=unit)
            target.write_text(json.dumps(result,indent=2))
            print(unit,result["total_population"],flush=True)
            return
        except OSError as e:
            if attempt==2:
                raise
            print(unit,str(e),flush=True)
            time.sleep(2**attempt)

if __name__=="__main__":
    features=json.loads((root/"gain_polygons.geojson").read_text())["features"]
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        list(pool.map(query,features))
