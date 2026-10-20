from pathlib import Path
import json,sys
import numpy as np
from replay_stable import read
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT.parent/"v9"),str(ROOT.parent/"v7")]
from run_match import evaluate_selected,write_csv
from benchmark import ious


def bind(box,boxes,ids):
    values=ious([box],boxes)[0];rank=np.argsort(values)
    if not len(rank) or values[rank[-1]]<.5:return None
    if len(rank)>1 and values[rank[-1]]-values[rank[-2]]<.1:return None
    return int(ids[rank[-1]])


def main():
    cfg=json.loads((ROOT/"second_holdout_selections.json").read_text());seq=cfg["sequence"]
    for backend in ["baseline","ocsort","deepocsort"]:
        assert (ROOT/"second_holdout"/backend/seq/"metrics.json").exists()
    gt=np.loadtxt(ROOT/"data"/seq/"gt/gt.txt",delimiter=",");g=gt[gt[:,0]==1]
    gb=g[:,2:6].copy();gb[:,2:]+=gb[:,:2];summary={}
    for target in cfg["targets"]:
        gid=bind(target["bbox_xyxy"],gb,g[:,1])
        assert gid is not None,"Seed is ambiguous in first-frame reference"
        summary[target["name"]]={}
        for backend in ["baseline","ocsort","deepocsort"]:
            out=ROOT/"second_holdout"/backend/seq;by=read(out/"tracks.csv")
            tid=bind(target["bbox_xyxy"],[[r[k] for k in ("x1","y1","x2","y2")] for r in by[1]],[r["id"] for r in by[1]])
            boxes=[]
            for frame in range(1,901):
                r=next((r for r in by[frame] if r["id"]==tid),None)
                boxes.append([r[k] for k in ("x1","y1","x2","y2")] if r else None)
            rows,m=evaluate_selected(boxes,gt,gid);m.update(first_frame_reference_id=gid,initial_model_id=tid)
            write_csv(out/f'{target["name"]}.csv',rows);summary[target["name"]][backend]=m
    (ROOT/"second_selection_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
