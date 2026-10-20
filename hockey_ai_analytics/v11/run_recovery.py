"""Replay cached causal detections with image-derived evidence; score separately."""
from pathlib import Path
import collections
import csv
import json
import sys
import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "v10"))
from appearance_tracker import descriptor
from recovery import Recovery
from score_recovery import evaluate


def read_tracks(path):
    by = collections.defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            by[int(r["frame"])].append(dict(id=int(r["id"]), cls=int(r["cls"]),
                box=[float(r[k]) for k in ("x1", "y1", "x2", "y2")]))
    return by


def predict(request, tracks, image_dir, config):
    engine = None
    outputs = []
    for frame in range(request["first_source_frame"], request["last_source_frame"] + 1):
        image = cv2.imread(str(image_dir / f"{frame:06d}.jpg"))
        if image is None:
            raise ValueError(f"Missing frame {frame}")
        if engine is None:
            engine = Recovery(config, request["seed_bbox_xyxy"],
                              descriptor(image, request["seed_bbox_xyxy"]), request["fps"])
        candidates = [{**r, "feature": descriptor(image, r["box"])} for r in tracks[frame]]
        outputs.append(dict(source_frame=frame, **engine.update(candidates)))
    return outputs


def score(folder, outputs, out):
    with (folder / "reference.csv").open() as f:
        refs = list(csv.DictReader(f))
    target = int(refs[0]["target_id"])
    reference = [dict(visible=r["visible"] == "True",
        box=[float(r[k]) for k in ("x1", "y1", "x2", "y2")] if r["visible"] == "True" else None) for r in refs]
    others = collections.defaultdict(list)
    with (folder / "all_reference.csv").open() as f:
        for r in csv.DictReader(f):
            if int(r["id"]) != target:
                others[int(r["frame"])].append([float(r[k]) for k in ("x1", "y1", "x2", "y2")])
    request = json.loads((folder / "request.json").read_text())
    rows, metrics = evaluate([r["box"] for r in outputs], reference, others, request["fps"])
    metrics.update(seed_bound=outputs[0]["box"] is not None,
                   recovery_emissions=sum(r["state"] == "RECOVERED" for r in outputs))
    with (out / "evaluation.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    config = json.loads((ROOT / "recovery_config.json").read_text())
    tracks = read_tracks(ROOT.parent / "v9/results/CHI_VS_TOR_004/tracks.csv")
    summary = []
    # Request manifests contain only seed information, no future labels.
    for path in sorted((ROOT / "cases").glob("*/request.json")):
        request = json.loads(path.read_text())
        out = ROOT / "results/recovery" / path.parent.name
        out.mkdir(parents=True, exist_ok=True)
        outputs = predict(request, tracks, ROOT.parent / "v9/data/CHI_VS_TOR_004/img1", config)
        (out / "predictions.json").write_text(json.dumps(outputs, indent=2))
        metrics = score(path.parent, outputs, out)
        summary.append(dict(case=path.parent.name, **metrics))
        print(path.parent.name, metrics["correct"], metrics["visible_frames"], metrics["wrong_person_frames"])
    (ROOT / "recovery_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
