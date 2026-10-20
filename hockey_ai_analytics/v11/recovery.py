"""Causal selected-player recovery. No labels, files, or future frames are read."""
import numpy as np


def iou(a, b):
    wh = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    area = wh.prod()
    return float(area / max(1e-9, np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - area))


def distance(a, b):
    if a is None or b is None:
        return 1.
    return float(np.sqrt(max(0., 1. - np.sqrt(a * b).sum())))


class Recovery:
    """Immutable seed appearance + motion gate + consecutive unique confirmation.

    Candidate IDs support short-term confirmation, never bypass appearance/motion.
    Missing/ambiguous observations produce UNKNOWN, with no retroactive filling.
    After expiry a new explicit selection is required.
    """
    def __init__(self, config, seed_box, seed_feature, fps):
        self.cfg = config
        self.seed = np.asarray(seed_box, float)
        self.anchor = seed_feature
        self.fps = fps
        self.last_frame = None
        self.last_box = None
        self.bound = None
        self.kind = None
        self.velocity = np.zeros(2)
        self.camera_since_accept = np.zeros(2)
        self.previous_candidates = {}
        self.pending = None
        self.pending_count = 0
        self.previous_output = False
        self.expired = False
        self.frame = 0

    def update(self, candidates):
        """Candidates: id, cls, box (xyxy), feature (normalized histogram)."""
        self.frame += 1
        boxes = [np.asarray(r["box"], float) for r in candidates]
        # Robust common translation is a proxy, not a calibrated camera model.
        shifts = []
        current = {}
        for r, b in zip(candidates, boxes):
            center = (b[:2] + b[2:]) / 2
            current[r["id"]] = center
            if r["id"] in self.previous_candidates:
                shift = center - self.previous_candidates[r["id"]]
                if np.linalg.norm(shift) < 40:
                    shifts.append(shift)
        camera = np.median(shifts, axis=0) if len(shifts) >= 4 else np.zeros(2)
        self.previous_candidates = current
        self.camera_since_accept += camera
        if self.frame == 1:
            ranks = sorted([(iou(self.seed, b), j) for j, b in enumerate(boxes)], reverse=True)
            if not ranks or ranks[0][0] < self.cfg["seed_iou"] or (
                len(ranks) > 1 and ranks[0][0] - ranks[1][0] < self.cfg["seed_margin"]
            ) or self.anchor is None:
                self.expired = True
                return self._unknown("SELECTION_FAILED")
            j = ranks[0][1]
            self.kind = candidates[j]["cls"]
            return self._accept(candidates[j], boxes[j], "SELECTED", camera)
        if self.expired:
            return self._unknown("RESELECT_REQUIRED")
        gap = self.frame - self.last_frame
        if gap > round(self.cfg["max_gap_seconds"] * self.fps):
            self.expired = True
            return self._unknown("RESELECT_REQUIRED")
        prediction = self.last_box + np.tile(
            self.camera_since_accept + self.velocity * min(gap, 5), 2)
        ranked = []
        for j, (r, box) in enumerate(zip(candidates, boxes)):
            if r["cls"] != self.kind:
                continue
            height = max(10., self.last_box[3] - self.last_box[1])
            ratio = max((box[3] - box[1]) / height, height / max(1., box[3] - box[1]))
            motion = np.linalg.norm((box[:2] + box[2:] - prediction[:2] - prediction[2:]) / 2) / height
            appearance = distance(self.anchor, r["feature"])
            radius = min(self.cfg["max_motion_radius"],
                         self.cfg["base_motion_radius"] + self.cfg["gap_motion_growth"] * (gap - 1))
            if ratio > self.cfg["max_scale_ratio"] or motion > radius or appearance > self.cfg["max_appearance_distance"]:
                continue
            ranked.append((.7 * appearance + .3 * motion / radius, j))
        ranked.sort()
        if not ranked:
            return self._unknown("NO_COMPATIBLE_CANDIDATE")
        if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < self.cfg["ambiguity_margin"]:
            return self._unknown("AMBIGUOUS")
        j = ranked[0][1]
        if any(k != j and iou(boxes[j], b) > self.cfg["max_overlap"] for k, b in enumerate(boxes)):
            return self._unknown("OCCLUDED")
        r, box = candidates[j], boxes[j]
        if r["id"] == self.bound and self.previous_output:
            return self._accept(r, box, "TRACKED", camera)
        self.pending_count = self.pending_count + 1 if self.pending == r["id"] else 1
        self.pending = r["id"]
        if self.pending_count < self.cfg["confirm_frames"]:
            self.previous_output = False
            return dict(box=None, state="CONFIRMING", candidate_id=r["id"])
        return self._accept(r, box, "RECOVERED", camera)

    def _unknown(self, reason):
        self.pending, self.pending_count, self.previous_output = None, 0, False
        return dict(box=None, state=reason, candidate_id=None)

    def _accept(self, candidate, box, state, camera):
        if self.last_box is not None:
            gap = self.frame - self.last_frame
            movement = (box[:2] + box[2:] - self.last_box[:2] - self.last_box[2:]) / 2
            self.velocity = .5 * self.velocity + .5 * (movement - self.camera_since_accept) / gap
        self.last_frame, self.last_box = self.frame, box.copy()
        self.bound = candidate["id"]
        self.camera_since_accept[:] = 0
        self.pending, self.pending_count, self.previous_output = None, 0, True
        return dict(box=box.tolist(), state=state, candidate_id=self.bound)
