from pathlib import Path
import sys,json,csv,argparse
import cv2,numpy as np
from replay_stable import read
from reciprocal_swap import ReciprocalSwap
from evaluation import evaluate
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"v7"))
from benchmark import ious


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--diagnostic-long",action="store_true");args=ap.parse_args()
    seq="CAR_VS_NYR_001" if args.diagnostic_long else "CHI_VS_TOR_004"
    n=1800 if args.diagnostic_long else 300
    out=ROOT/"swap"/seq;out.mkdir(parents=True,exist_ok=True)
    detroot=ROOT/"long_baseline" if args.diagnostic_long else ROOT.parent/"v9/results"
    samroot=ROOT/"long_candidate" if args.diagnostic_long else ROOT/"multisam"
    det=read(detroot/seq/"tracks.csv")
    sam=read(samroot/seq/"sam_tracks.csv")
    seeds=json.loads((samroot/seq/"seeds.json").read_text())
    data=(ROOT/"data" if args.diagnostic_long else ROOT.parent/"v9/data")/seq
    model=ReciprocalSwap(seeds,cv2.imread(str(data/"img1/000001.jpg")),ious)
    rows=[]
    for frame in range(1,n+1):
        im=cv2.imread(str(data/"img1"/f"{frame:06d}.jpg"))
        rows.extend(model.update(frame,im,det[frame],sam[frame]))
    with (out/"tracks.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    gt=np.loadtxt(data/"gt/gt.txt",delimiter=",")
    metrics=evaluate(rows,gt,n)
    metrics["validation_scope"]="reused holdout diagnostic, not fresh validation" if args.diagnostic_long else "development"
    metrics["changed_frames"]=len({r["frame"] for r in rows if r["association_state"]=="RECIPROCAL_SWAP"})
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics))


if __name__=="__main__":main()
