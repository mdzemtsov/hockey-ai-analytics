from pathlib import Path
import json, csv, sys, collections
import cv2
import numpy as np
from appearance_tracker import AppearanceTracker

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/"v7"))
from benchmark import score


def main():
    config = json.loads((ROOT/"appearance_protocol.json").read_text())
    summary = {}
    for seq in config["sequences"]:
        source = ROOT.parent/"v9/results"/seq
        out = ROOT/"appearance"/seq
        out.mkdir(parents=True, exist_ok=True)
        by = collections.defaultdict(list)
        with (source/"tracks.csv").open() as f:
            for r in csv.DictReader(f):
                r = {k: float(v) for k, v in r.items()}
                for k in ("frame", "id", "cls"):
                    r[k] = int(r[k])
                by[r["frame"]].append(r)
        tracker = AppearanceTracker(config)
        predictions = []
        for frame in range(1, config["frames"]+1):
            im = cv2.imread(str(ROOT.parent/"v9/data"/seq/"img1"/f"{frame:06d}.jpg"))
            predictions.extend(tracker.update(frame, im, by[frame]))
        with (out/"tracks.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        # Future annotations first read AFTER output saved.
        gt = np.loadtxt(ROOT.parent/"v9/data"/seq/"gt/gt.txt", delimiter=",")
        result = dict(baseline=score([r for v in by.values() for r in v], gt),
                      appearance=score(predictions, gt))
        (out/"metrics.json").write_text(json.dumps(result, indent=2))
        summary[seq] = result
        print(seq, json.dumps(result), flush=True)
    (ROOT/"appearance_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
