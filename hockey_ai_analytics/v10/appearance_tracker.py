"""Causal association of cached detections; no annotation access."""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def descriptor(image, box):
    x1, y1, x2, y2 = np.asarray(box, float)
    w, h = x2-x1, y2-y1
    H, W = image.shape[:2]
    l, r = np.clip([x1+.2*w, x2-.2*w], 0, W).astype(int)
    t, b = np.clip([y1+.15*h, y1+.55*h], 0, H).astype(int)
    crop = image[t:b, l:r]
    if crop.size < 60:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hue, sat, val = hsv.T.astype(int)
    # Achromatic pixels need separate dark/grey/white bins, not arbitrary hue.
    bins = np.where(sat < 55, 24 + np.minimum(val // 32, 7),
                    np.minimum(hue // 15, 11) + 12 * (val >= 128))
    hist = np.bincount(bins, minlength=32).astype(float)
    return hist / hist.sum()


def distance(a, b):
    if a is None or b is None:
        return .5
    return float(np.sqrt(max(0, 1-np.sqrt(a*b).sum())))


def overlap(a, b):
    wh = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]))
    inter = wh.prod()
    return inter/max(1e-9, (a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-inter)


class AppearanceTracker:
    def __init__(self, config):
        self.cfg = config
        self.tracks = {}
        self.next_id = 1

    def update(self, frame, image, detections):
        boxes = [np.array([r[k] for k in ("x1", "y1", "x2", "y2")]) for r in detections]
        feats = [descriptor(image, b) for b in boxes]
        self.tracks = {i: t for i, t in self.tracks.items()
                       if frame-t["frame"] <= self.cfg["max_age_frames"]}
        ids = list(self.tracks)
        cost = np.full((len(ids), len(boxes)), 1e6)
        for i, tid in enumerate(ids):
            t = self.tracks[tid]
            dt = frame-t["frame"]
            pred = t["box"] + np.tile(t["velocity"]*min(dt, 3), 2)
            for j, box in enumerate(boxes):
                if (t["cls"] == 6) != (detections[j]["cls"] == 6):
                    continue
                scale = max(10, (pred[3]-pred[1]+box[3]-box[1])/2)
                center = np.linalg.norm((pred[:2]+pred[2:]-box[:2]-box[2:])/2)/scale
                appearance = .5*distance(t["anchor"], feats[j])+.5*distance(t["appearance"], feats[j])
                if center > self.cfg["max_normalized_center_distance"] or appearance > self.cfg["max_appearance_distance"]:
                    continue
                cost[i, j] = .45*(1-overlap(pred, box))+.35*appearance+.2*min(center, 1)
        assigned = {}
        if cost.size:
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                best = cost[i, j]
                alternatives = np.r_[np.delete(cost[i], j), np.delete(cost[:, j], i)]
                margin = float(alternatives.min()-best) if len(alternatives) else 1
                if best <= self.cfg["max_assignment_cost"] and margin >= self.cfg["ambiguity_margin"]:
                    assigned[j] = (ids[i], float(best), margin)
        result = []
        for j, (r, box, feat) in enumerate(zip(detections, boxes, feats)):
            match = assigned.get(j)
            if match is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = dict(box=box, frame=frame, cls=r["cls"],
                                        velocity=np.zeros(2), anchor=feat, appearance=feat)
                state, c, margin = "NEW_OR_UNCERTAIN", None, None
            else:
                tid, c, margin = match
                t = self.tracks[tid]
                velocity = ((box[:2]+box[2:]-t["box"][:2]-t["box"][2:])/2)/(frame-t["frame"])
                t["velocity"] = .5*t["velocity"]+.5*velocity
                # Avoid learning mixed crops during close encounters.
                isolated = all(k == j or overlap(box, other) < .1 for k, other in enumerate(boxes))
                if isolated and feat is not None:
                    t["appearance"] = feat if t["appearance"] is None else .9*t["appearance"]+.1*feat
                t.update(box=box, frame=frame)
                state = "ASSOCIATED"
            result.append({**r, "id": tid, "input_id": r["id"], "association_state": state,
                           "association_cost": c, "association_margin": margin})
        return result
