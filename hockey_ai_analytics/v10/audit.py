"""Evaluation-only identity event audit. Never imported by inference."""
from pathlib import Path
import sys, json, csv
import numpy as np
import cv2
import motmetrics as mm

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE / "v7"))
from benchmark import ious


def audit(seq, variant="baseline"):
    out = ROOT / ("audit" if variant == "baseline" else "audit_stable") / seq
    out.mkdir(parents=True, exist_ok=True)
    source = BASE / "v9/results" / seq / "tracks.csv" if variant == "baseline" else ROOT / "stable" / seq / "tracks.csv"
    with source.open() as f:
        rows = [{k: float(r[k]) for k in ("frame","id","x1","y1","x2","y2","cls","confidence")}
                for r in csv.DictReader(f)]
    gt = np.loadtxt(BASE / "v9/data" / seq / "gt/gt.txt", delimiter=",")
    acc = mm.MOTAccumulator(auto_id=True)
    for frame in range(1, 301):
        g = gt[gt[:, 0] == frame]
        p = [r for r in rows if r["frame"] == frame]
        gb = g[:, 2:6].copy()
        gb[:, 2:] += gb[:, :2]
        d = 1 - ious(gb, [[r[k] for k in ("x1", "y1", "x2", "y2")] for r in p])
        d[d > .5] = np.nan
        acc.update(g[:, 1].astype(int), [int(r["id"]) for r in p], d)
    events = acc.mot_events.reset_index()
    events["source_frame"] = events["FrameId"] + 1
    events.to_csv(out / "mot_events.csv", index=False)
    switches = events[events.Type == "SWITCH"]
    cases = []
    for _, e in switches.iterrows():
        frame, gid, new = int(e.source_frame), int(e.OId), int(e.HId)
        prior = events[(events.source_frame < frame) & (events.OId == gid)
                       & events.Type.isin(["MATCH", "SWITCH"])]
        old = int(prior.iloc[-1].HId) if len(prior) else None
        last = int(prior.iloc[-1].source_frame) if len(prior) else None
        case = dict(frame=frame, reference_id=gid, old_model_id=old, new_model_id=new,
                    last_matched_frame=last, gap_frames=frame-last-1 if last else None,
                    disposition="UNREVIEWED", cause="UNKNOWN")
        cases.append(case)
        panels = []
        for fr in sorted(set([max(1, frame-5), last or frame, frame, min(300, frame+5)])):
            im = cv2.imread(str(BASE / "v9/data" / seq / "img1" / f"{fr:06d}.jpg"))
            for r in rows:
                if r["frame"] != fr or int(r["id"]) not in (old, new):
                    continue
                box = np.array([r[k] for k in ("x1", "y1", "x2", "y2")], int)
                color = (0, 255, 255) if int(r["id"]) == new else (255, 190, 0)
                cv2.rectangle(im, tuple(box[:2]), tuple(box[2:]), color, 3)
                cv2.putText(im, f"model {int(r['id'])}", tuple(box[:2]-[0, 7]),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
            for g in gt[(gt[:, 0] == fr) & (gt[:, 1] == gid)]:
                x, y, w, h = g[2:6].astype(int)
                cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 255), 2)
            cv2.putText(im, f"frame {fr} REF {gid} (magenta)", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
            panels.append(cv2.resize(im, (640, 360)))
        while len(panels) < 4:
            panels.append(np.zeros((360, 640, 3), np.uint8))
        cv2.imwrite(str(out / f"switch_{frame:04d}_ref{gid}.jpg"),
                    np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:4])]))
    (out / "switch_cases.json").write_text(json.dumps(cases, indent=2))
    summary = dict(sequence=seq, switches=len(cases),
                   immediate=sum(c["gap_frames"] == 0 for c in cases),
                   after_gap=sum(c["gap_frames"] > 0 for c in cases),
                   note="Metric association events, not yet visually confirmed identity errors.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    for seq in sys.argv[1:] or ["CHI_VS_TOR_004", "CAR_VS_BOS_001", "CGY_VS_DAL_003", "STL_VS_SJ_001"]:
        audit(seq)
