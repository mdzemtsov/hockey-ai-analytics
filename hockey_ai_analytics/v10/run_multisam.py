from pathlib import Path
import sys, json, csv, collections, time, argparse
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics.models.sam import SAM2VideoPredictor
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/"v7"))
from benchmark import ious, score


def write(path, rows):
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--deduplicate-seeds",action="store_true")
    args=ap.parse_args()
    cfg = json.loads((ROOT/"multisam_protocol.json").read_text())
    seq = cfg["sequence"]
    source = ROOT.parent/"v9/results"/seq
    out = ROOT/("multisam_nms" if args.deduplicate_seeds else "multisam")/seq
    out.mkdir(parents=True, exist_ok=True)
    by = collections.defaultdict(list)
    with (source/"tracks.csv").open() as f:
        for r in csv.DictReader(f):
            r = {k: float(v) for k,v in r.items()}
            for k in ("frame","id","cls"):
                r[k] = int(r[k])
            by[r["frame"]].append(r)
    seeds = [r for r in by[1] if r["confidence"] >= .25]
    if args.deduplicate_seeds:
        kept=[]
        for r in sorted(seeds,key=lambda r:-r["confidence"]):
            b=[r[k] for k in ("x1","y1","x2","y2")]
            same=[s for s in kept if s["cls"]==r["cls"]]
            overlaps=ious([b],[[s[k] for k in ("x1","y1","x2","y2")] for s in same])
            if overlaps.size and overlaps.max()>.5:
                continue
            kept.append(r)
        seeds=kept
    (out/"seeds.json").write_text(json.dumps(seeds, indent=2))
    scale = np.array([1280/848, 720/477]*2)
    boxes = [(np.array([r[k] for k in ("x1","y1","x2","y2")])/scale).tolist() for r in seeds]
    torch.set_num_threads(4)
    predictor = SAM2VideoPredictor(overrides=dict(
        model=str(ROOT.parent/"v8/models/sam2.1_t.pt"), imgsz=1024,
        device="cpu", save=False, verbose=False, conf=.25))
    raw, combined = [], []
    start = time.time()
    for frame, p in enumerate(predictor(source=str(source/"input.mp4"), bboxes=boxes, stream=True),1):
        current = []
        # cls retains ORIGINAL prompt index even when absent masks filtered.
        for b, idx in zip(p.boxes.xyxy.cpu().numpy(), p.boxes.cls.int().tolist()):
            b = b*scale
            if b[2] <= b[0] or b[3] <= b[1]:
                continue
            assert 0 <= idx < len(seeds)
            current.append(dict(frame=frame,id=1000+idx,x1=float(b[0]),y1=float(b[1]),
                                x2=float(b[2]),y2=float(b[3]),confidence=1.,cls=seeds[idx]["cls"]))
        raw.extend(current)
        det = by[frame]
        ov = ious([[r[k] for k in ("x1","y1","x2","y2")] for r in current],
                  [[r[k] for k in ("x1","y1","x2","y2")] for r in det])
        matches = {}
        if ov.size:
            ri, ci = linear_sum_assignment(1-ov)
            for i,j in zip(ri,ci):
                alternatives = np.r_[np.delete(ov[i],j),np.delete(ov[:,j],i)]
                second = alternatives.max() if len(alternatives) else 0
                if ov[i,j]>=.5 and ov[i,j]-second>=.15:
                    matches[j] = current[i]["id"]
        for j,r in enumerate(det):
            combined.append({**r,"id":matches.get(j,10000+r["id"]),
                             "source":"SAM_CONFIRMED" if j in matches else "BASELINE_FALLBACK"})
        if frame % 25 == 0:
            print(seq,frame,"elapsed",round(time.time()-start,1),flush=True)
    assert frame == cfg["frames"]
    write(out/"sam_tracks.csv",raw)
    write(out/"combined_tracks.csv",combined)
    gt = np.loadtxt(ROOT.parent/"v9/data"/seq/"gt/gt.txt",delimiter=",")
    metrics = dict(baseline=score([r for v in by.values() for r in v],gt),
                   sam=score(raw,gt),combined=score(combined,gt),
                   seconds=time.time()-start,seed_count=len(seeds))
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics),flush=True)


if __name__ == "__main__":
    main()
