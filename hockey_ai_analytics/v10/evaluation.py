"""Frame-count-explicit evaluation, independent of the old 300-frame helper."""
import numpy as np
import motmetrics as mm
from collections import defaultdict


def evaluate(rows, gt, frame_count):
    if frame_count < 1:
        raise ValueError("Positive frame_count required")
    by = defaultdict(list)
    for r in rows:
        f = int(r["frame"])
        if f < 1 or f > frame_count:
            raise ValueError("Prediction outside evaluation interval")
        by[f].append(r)
    acc = mm.MOTAccumulator(auto_id=True)
    for f in range(1, frame_count+1):
        g = gt[gt[:,0] == f]
        p = by[f]
        ids = [int(r["id"]) for r in p]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate identity in frame")
        gb = g[:,2:6].copy()
        gb[:,2:] += gb[:,:2]
        pb = np.asarray([[r[k] for k in ("x1","y1","x2","y2")] for r in p]).reshape(-1,4)
        lo = np.maximum(gb[:,None,:2],pb[None,:,:2])
        hi = np.minimum(gb[:,None,2:],pb[None,:,2:])
        inter = np.maximum(0,hi-lo).prod(2)
        union = (gb[:,2:]-gb[:,:2]).prod(1)[:,None]+(pb[:,2:]-pb[:,:2]).prod(1)[None,:]-inter
        d = 1-inter/np.maximum(union,1e-9)
        d[d>.5] = np.nan
        acc.update(g[:,1].astype(int),ids,d)
    metrics = ["num_frames","num_objects","num_predictions","num_matches","num_misses",
               "num_false_positives","num_switches","precision","recall","idf1","mota"]
    return {k:float(v) for k,v in mm.metrics.create().compute(acc,metrics=metrics).iloc[0].items()}
