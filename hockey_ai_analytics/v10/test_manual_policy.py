from pathlib import Path
import json,csv,sys,collections,configparser
import numpy as np
from reconfirmation_gate import ReconfirmationGate,ManualGate
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"v9"))
from run_match import evaluate_selected,write_csv


def main():
    selections=json.loads((ROOT.parent/"v9/selections.json").read_text());summary={}
    scale=np.array([848/1280,477/720]*2)
    for seq,s in selections.items():
        data=ROOT.parent/"v9/data"/seq;out=ROOT/"manual_policy"/seq;out.mkdir(parents=True,exist_ok=True)
        ini=configparser.ConfigParser();ini.read(data/"seqinfo.ini");fps=ini.getint("Sequence","frameRate")
        by=collections.defaultdict(list)
        with (ROOT.parent/"v9/results"/seq/"tracks.csv").open() as f:
            for r in csv.DictReader(f):
                x1,y1,x2,y2=np.array([float(r[k]) for k in ("x1","y1","x2","y2")])*scale
                by[int(r["frame"])].append(dict(track_id=int(r["id"]),x=x1,y=y1,w=x2-x1,h=y2-y1,
                                                detector_conf=float(r["confidence"]),team_hypothesis="UNKNOWN"))
        b=np.array(s["bbox_xyxy"])*scale
        anchor=dict(frame_index=0,bbox=[b[0],b[1],b[2]-b[0],b[3]-b[1]])
        predictions={};logs={}
        for name,cls in [("auto_recovery",ManualGate),("explicit_reconfirmation",ReconfirmationGate)]:
            gate=cls([anchor]);rows=[];boxes=[]
            for frame in range(1,301):
                row=gate.update(frame-1,(frame-1)/fps,by[frame]);rows.append(row)
                if row["state"]=="TRACKING":
                    bb=np.array([row["x"],row["y"],row["x"]+row["w"],row["y"]+row["h"]])/scale
                    boxes.append(bb.tolist())
                else:boxes.append(None)
            predictions[name]=boxes;logs[name]=rows
            write_csv(out/f"{name}_states.csv",rows)
        gt=np.loadtxt(data/"gt/gt.txt",delimiter=",");metrics={}
        for name,boxes in predictions.items():
            rows,m=evaluate_selected(boxes,gt,s["published_id"])
            m["first_nontracking_frame"]=next((i+1 for i,r in enumerate(logs[name]) if r["state"]!="TRACKING"),None)
            write_csv(out/f"{name}_evaluation.csv",rows);metrics[name]=m
        (out/"metrics.json").write_text(json.dumps(metrics,indent=2));summary[seq]=metrics
    (ROOT/"manual_policy_summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
