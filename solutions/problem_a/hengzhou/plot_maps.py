"""Draw inundation changes and assessment priorities from the model outputs."""
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root = Path.cwd()

def map_font(names, size):
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)

def make_maps():
    with (root / "q4_risk.csv").open(encoding="utf-8-sig") as source:
        risk = list(csv.DictReader(source))
    fnt = map_font(("arial.ttf", "DejaVuSans.ttf"), 22)
    bold = map_font(("arialbd.ttf", "DejaVuSans-Bold.ttf"), 23)
    with Image.open(root/"July06_transitions.tif") as im:
        arr=im.convert("L")
        lut=[(242,240,229)]*256
        for k,color in [(1,(0,130,150)),(2,(204,105,60)),(3,(35,69,105)),(255,(173,178,180))]:lut[k]=color
        arr=arr.convert("P")
        arr.putpalette([a for color in lut for a in color]);arr=arr.convert("RGB")
    img=Image.new("RGB",(1590,572),"white");img.paste(arr,(42,52));d=ImageDraw.Draw(img)
    d.text((42,12),"109.2200 E",font=fnt,fill="#42515a");d.text((1410,12),"109.3625 E",font=fnt,fill="#42515a")
    d.text((740,12),"22.9200 N",font=fnt,fill="#42515a")
    for x in range(7):d.line((42+x*252,52,42+x*252,476),fill="#697a7c",width=2)
    for y in range(3):d.line((42,52+y*212,1554,52+y*212),fill="#697a7c",width=2)
    for r in range(2):
        for c in range(6):
            x=48+252*c;y=60+212*r
            d.rectangle((x,y,x+62,y+28),fill="white");d.text((x+3,y+1),f"r{r+5}c{c+1}",font=fnt,fill="#1f333c")
    d.text((740,482),"22.8800 N",font=fnt,fill="#42515a")
    labels=[("Dry",lut[0]),("New water",lut[1]),("Water loss",lut[2]),("Persistent water",lut[3]),("Obscured",lut[255])]
    for i,(s,col) in enumerate(labels):
        x=52+i*296;d.rectangle((x,534,x+27,555),fill=col,outline="#829197");d.text((x+38,529),s,font=fnt,fill="#20323c")
    img.save(root/"Q3_event_map.png")

    img=Image.new("RGB",(1512,396),"white");d=ImageDraw.Draw(img)
    for item in risk:
        r,c=map(int,item["unit"][1:].split("c"));x=(c-1)*252;y=(r-5)*194
        z=float(item["score"])/100
        colour=tuple(round(a+(b-a)*z) for a,b in zip((243,246,240),(21,96,104)))
        d.rectangle((x+2,y+2,x+248,y+190),fill=colour)
        ink="white" if z>.64 else "#123239"
        d.text((x+14,y+15),item["unit"]+"  /  #"+item["rank"],font=bold,fill=ink)
        d.text((x+14,y+65),f"Score   {float(item['score']):.1f}",font=fnt,fill=ink)
        d.text((x+14,y+110),f"Exposed   {float(item['exposed']):,.0f}",font=fnt,fill=ink)
    img.save(root/"Q4_priority_map.png")

if __name__ == "__main__":
    make_maps()
