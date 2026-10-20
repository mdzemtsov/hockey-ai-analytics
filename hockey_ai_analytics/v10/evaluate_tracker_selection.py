"""Same previously selected first-frame player, no automatic identity rebinding."""
from pathlib import Path
import json,sys
import numpy as np
from replay_stable import read
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT.parent/"v9"),str(ROOT.parent/"v7")]
from run_match import evaluate_selected,write_csv
from benchmark import ious


def main():
    selections=json.loads((ROOT.parent/"v9/selections.json").read_text())
    summary={}
    for backend in ["ocsort","deepocsort"]:
        summary[backend]={}
        for seq,s in selections.items():
            out=ROOT/"trackers"/backend/seq
            by=read(out/"tracks.csv");first=by[1]
            ov=ious([s["bbox_xyxy"]],[[r[k] for k in ("x1","y1","x2","y2")] for r in first])[0]
            bound=first[int(ov.argmax())]["id"] if len(ov) and ov.max()>=.5 else None
            boxes=[]
            for frame in range(1,301):
                r=next((r for r in by[frame] if r["id"]==bound),None)
                boxes.append([r[k] for k in ("x1","y1","x2","y2")] if r else None)
            gt=np.loadtxt(ROOT.parent/"v9/data"/seq/"gt/gt.txt",delimiter=",")
            rows,m=evaluate_selected(boxes,gt,s["published_id"]);m["initial_model_id"]=bound
            write_csv(out/"selected.csv",rows)
            (out/"selected_metrics.json").write_text(json.dumps(m,indent=2))
            summary[backend][seq]=m
    (ROOT/"tracker_selection_summary.json").write_text(json.dumps(summary,indent=2))
    for b,seqs in summary.items():
        for s,m in seqs.items():print(b,s,m["correct"],m["matches_other_identity"],m["visible_lost"])


if __name__=="__main__":main()
