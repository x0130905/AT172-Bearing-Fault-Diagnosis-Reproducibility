"""Regenerate the two evidence figures from retained metrics and histories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from PIL import Image, ImageDraw, ImageFont


METHODS = ["baseline", "coral", "mixup", "mmd"]
LABELS = ["ERM", "CORAL", "Mixup", "Linear MMD"]
SEEDS = [20260916, 20260917, 20260918, 42, 123]


def get_font(size, bold=False):
    candidates = [
        Path("C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def macro_f1(root, method, seed):
    return 100 * json.loads((root / "metrics" / f"{method}_{seed}" / "metrics.json").read_text(encoding="utf-8"))["unseen_test_3"]["macro_f1"]


def figure1(root, output):
    image = Image.new("RGB", (1200, 555), "white"); draw = ImageDraw.Draw(image)
    axis, tick, legend = get_font(35), get_font(31), get_font(31)
    left, top, right, bottom = 154, 108, 1160, 452; ymin, ymax = 48, 103
    ypx = lambda value: bottom - (value-ymin)/(ymax-ymin)*(bottom-top)
    draw.line((left,top,left,bottom),fill="black",width=4); draw.line((left,bottom,right,bottom),fill="black",width=4)
    for value in (50,60,70,80,90,100):
        y=ypx(value); draw.line((left,y,right,y),fill="#d2d2d2",width=2); draw.text((108,y-16),str(value),font=tick,fill="black")
    for x,method,label in zip((250,500,750,1000),METHODS,LABELS):
        values=[macro_f1(root,method,seed) for seed in SEEDS]
        for offset,value in zip((-34,-17,0,17,34),values):
            y=ypx(value); draw.ellipse((x+offset-8,y-8,x+offset+8,y+8),fill="white",outline="black",width=3)
        m,sd=mean(values),stdev(values); y0,y1,ym=ypx(m-sd),ypx(m+sd),ypx(m)
        draw.line((x,y0,x,y1),fill="black",width=4); draw.line((x-16,y0,x+16,y0),fill="black",width=4); draw.line((x-16,y1,x+16,y1),fill="black",width=4); draw.line((x-22,ym,x+22,ym),fill="black",width=5)
        box=draw.textbbox((0,0),label,font=tick); draw.text((x-(box[2]-box[0])/2,bottom+15),label,font=tick,fill="black")
    ylabel=Image.new("RGBA",(430,48),(255,255,255,0)); ImageDraw.Draw(ylabel).text((0,0),"Unseen 3-hp Macro-F1 (%)",font=axis,fill="black"); ylabel=ylabel.rotate(90,expand=True); image.paste(ylabel,(20,64),ylabel)
    draw.rectangle((578,22,1158,91),fill="white",outline="#b0b0b0",width=2); draw.ellipse((604,47,620,63),fill="white",outline="black",width=3); draw.text((635,35),"individual seed",font=legend,fill="black"); draw.line((873,55,921,55),fill="black",width=5); draw.text((937,35),"mean +/- SD",font=legend,fill="black")
    image.save(output,dpi=(360,360))


def figure2(root, output):
    image=Image.new("RGB",(1200,720),"white"); draw=ImageDraw.Draw(image); axis,tick,legend=get_font(32),get_font(29),get_font(31)
    left,top,right,bottom=155,46,1160,602; xpx=lambda e:left+(e-1)/28*(right-left); ypx=lambda v:bottom-v/104*(bottom-top)
    draw.line((left,top,left,bottom),fill="black",width=4); draw.line((left,bottom,right,bottom),fill="black",width=4)
    for value in (0,25,50,75,100):
        y=ypx(value); draw.line((left,y,right,y),fill="#d2d2d2",width=2); draw.text((105,y-16),str(value),font=tick,fill="black")
    for epoch in (1,5,10,15,20,25,29):
        x=xpx(epoch); draw.line((x,bottom,x,bottom+9),fill="black",width=3); label=str(epoch); box=draw.textbbox((0,0),label,font=tick); draw.text((x-(box[2]-box[0])/2,bottom+14),label,font=tick,fill="black")
    for seed,color,square,dashed in ((42,"#111111",False,False),(20260918,"#666666",True,True)):
        rows=json.loads((root/"histories"/f"baseline_{seed}"/"history.json").read_text(encoding="utf-8")); points=[(xpx(r["epoch"]),ypx(100*r["validation_macro_f1"])) for r in rows]
        if dashed:
            for i,(a,b) in enumerate(zip(points[:-1],points[1:])):
                draw.line((a,b),fill=color,width=5)
        else: draw.line(points,fill=color,width=5)
        for x,y in points:
            if square: draw.rectangle((x-7,y-7,x+7,y+7),outline=color,fill="white",width=3)
            else: draw.ellipse((x-7,y-7,x+7,y+7),outline=color,fill="white",width=3)
    draw.text((615,674),"Epoch",font=axis,fill="black"); ylabel=Image.new("RGBA",(430,48),(255,255,255,0)); ImageDraw.Draw(ylabel).text((0,0),"Validation Macro-F1 (%)",font=axis,fill="black"); ylabel=ylabel.rotate(90,expand=True); image.paste(ylabel,(22,136),ylabel)
    draw.rectangle((620,57,1149,150),fill="white",outline="#b0b0b0",width=2); draw.line((638,87,710,87),fill="#111111",width=5); draw.ellipse((669,80,683,94),outline="#111111",fill="white",width=3); draw.text((725,69),"ERM seed 42 (weak)",font=legend,fill="black"); draw.line((638,124,710,124),fill="#666666",width=5); draw.rectangle((669,117,683,131),outline="#666666",fill="white",width=3); draw.text((725,106),"ERM seed 20260918 (strong)",font=legend,fill="black")
    image.save(output,dpi=(360,360))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--artifact-root",type=Path,default=Path(__file__).resolve().parents[1]); parser.add_argument("--output-dir",type=Path,required=True); args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    figure1(args.artifact_root,args.output_dir/"fig1_method_variability.png"); figure2(args.artifact_root,args.output_dir/"fig2_validation_trajectories.png"); print(args.output_dir)


if __name__=="__main__": main()
