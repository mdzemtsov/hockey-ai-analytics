from pathlib import Path
import sys,json,csv,collections,argparse
import numpy as np
from stable_fusion import StableFusion
from evaluation import evaluate
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"v7"))
from benchmark import ious


def read(path):
    by=collections.defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            r={k:float(v) for k,v in r.items()}
            for k in ("frame","id","cls"):r[k]=int(r[k])
            by[r["frame"]].append(r)
    return by


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--deduplicate-seeds",action="store_true");args=ap.parse_args()
    folder="multisam_nms" if args.deduplicate_seeds else "multisam"
    seq="CHI_VS_TOR_004";out=ROOT/("stable_nms" if args.deduplicate_seeds else "stable")/seq;out.mkdir(parents=True,exist_ok=True)
    det=read(ROOT.parent/"v9/results"/seq/"tracks.csv")
    sam=read(ROOT/folder/seq/"sam_tracks.csv")
    seeds=json.loads((ROOT/folder/seq/"seeds.json").read_text())
    fusion=StableFusion([r["id"] for r in seeds],ious)
    rows=[]
    for frame in range(1,301):rows.extend(fusion.update(frame,det[frame],sam[frame]))
    with (out/"tracks.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    gt=np.loadtxt(ROOT.parent/"v9/data"/seq/"gt/gt.txt",delimiter=",")
    metrics=evaluate(rows,gt,300)
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics))


if __name__=="__main__":main()
