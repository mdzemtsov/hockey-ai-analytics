"""Frozen-candidate temporal validation; labels read only after saving predictions."""
from pathlib import Path
import csv
import hashlib
import json
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from run_recovery import read_tracks, predict, score
from score_recovery import iou

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reserved"


def write(path, rows):
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    lock = json.loads((ROOT / "candidate_lock.json").read_text())
    for entry in lock["files"]:
        assert hashlib.sha256((ROOT.parent / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    manifest = json.loads((OUT / "source_manifest.json").read_text())
    assert len(manifest) == 600
    for entry in manifest:
        assert hashlib.sha256((OUT / "img1" / f"{entry['frame']:06d}.jpg").read_bytes()).hexdigest() == entry["sha256"]
    track_path = OUT / "tracks.csv"
    if not track_path.exists():
        torch.set_num_threads(4)
        model = YOLO(str(ROOT.parent / "v7/data/HockeyAI.pt"))
        rows = []
        start = time.time()
        for frame in range(901, 1501):
            im = cv2.imread(str(OUT / "img1" / f"{frame:06d}.jpg"))
            p = model.track(cv2.resize(im, (848, 477)), persist=True, classes=[3, 4, 6],
                conf=.1, iou=.7, imgsz=960, tracker=str(ROOT.parent / "v5/botsort_reid.yaml"),
                verbose=False, device="cpu")[0]
            if p.boxes.id is not None:
                for box, tid, cls in zip(p.boxes.xyxy.cpu().numpy(), p.boxes.id.int().tolist(), p.boxes.cls.int().tolist()):
                    b = box * np.array([1280 / 848, 720 / 477] * 2)
                    rows.append(dict(frame=frame, id=tid, cls=cls, x1=b[0], y1=b[1], x2=b[2], y2=b[3]))
            if frame % 50 == 0:
                print(f"inference {frame-900}/600 {time.time()-start:.1f}s", flush=True)
        write(track_path, rows)
    tracks = read_tracks(track_path)
    config = json.loads((ROOT / "recovery_config.json").read_text())
    seeds = json.loads((OUT / "selections.json").read_text())["targets"]
    all_predictions = {}
    for seed in seeds:
        folder = OUT / seed["name"]
        folder.mkdir(exist_ok=True)
        request = dict(first_source_frame=901, last_source_frame=1500, fps=30,
                       seed_bbox_xyxy=seed["bbox_xyxy"])
        (folder / "request.json").write_text(json.dumps(request, indent=2))
        recovery = predict(request, tracks, OUT / "img1", config)
        ranked = sorted([(iou(seed["bbox_xyxy"], r["box"]), r["id"]) for r in tracks[901]], reverse=True)
        bound = ranked[0][1] if ranked and ranked[0][0] >= .5 and (
            len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= .1) else None
        baseline = []
        for frame in range(901, 1501):
            row = next((r for r in tracks[frame] if r["id"] == bound), None)
            baseline.append(dict(source_frame=frame, box=row["box"] if row else None,
                                 state="FIXED_ID" if row else "UNKNOWN"))
        all_predictions[seed["name"]] = dict(baseline=baseline, recovery=recovery)
        for method, predictions in all_predictions[seed["name"]].items():
            output = folder / method
            output.mkdir(exist_ok=True)
            (output / "predictions.json").write_text(json.dumps(predictions, indent=2))
    # Both methods' complete predictions have been persisted before reference loading.
    gt_path = ROOT.parent / "v10/data/CNTRL_VS_PAC_001/gt/gt.txt"
    gt = np.loadtxt(gt_path, delimiter=",")
    summary = []
    for seed in seeds:
        folder = OUT / seed["name"]
        first = gt[gt[:, 0] == 901]
        ranked = sorted([(iou(seed["bbox_xyxy"], [r[2], r[3], r[2]+r[4], r[3]+r[5]]), int(r[1])) for r in first], reverse=True)
        assert ranked and ranked[0][0] >= .5 and (len(ranked) == 1 or ranked[0][0]-ranked[1][0] >= .1), "Reference seed binding ambiguous"
        target = ranked[0][1]
        refs, others = [], []
        for frame in range(901, 1501):
            present = gt[gt[:, 0] == frame]
            target_rows = present[present[:, 1] == target]
            ref = dict(frame=frame-900, source_frame=frame, target_id=target,
                       visible=bool(len(target_rows)), x1="", y1="", x2="", y2="")
            if len(target_rows):
                x,y,w,h = target_rows[0,2:6]
                ref.update(x1=x,y1=y,x2=x+w,y2=y+h)
            refs.append(ref)
            for r in present:
                x,y,w,h = r[2:6]
                others.append(dict(frame=frame-900,id=int(r[1]),x1=x,y1=y,x2=x+w,y2=y+h))
        write(folder / "reference.csv", refs)
        write(folder / "all_reference.csv", others)
        for method, predictions in all_predictions[seed["name"]].items():
            metrics = score(folder, predictions, folder / method)
            summary.append(dict(target=seed["name"], method=method, **metrics))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "reference_provenance.json").write_text(json.dumps(dict(
        source=str(gt_path),sha256=hashlib.sha256(gt_path.read_bytes()).hexdigest(),
        independent_human_verified=False, future_reference_used_in_inference=False),indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
