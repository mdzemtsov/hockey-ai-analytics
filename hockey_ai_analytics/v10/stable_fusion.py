"""Keep identity across temporary SAM disagreement; confirmations are causal."""
import numpy as np
from scipy.optimize import linear_sum_assignment


class StableFusion:
    def __init__(self, seed_ids, iou_function, persistence=3):
        self.seed_ids=seed_ids
        self.iou=iou_function
        self.persistence=persistence
        self.alias={}
        self.pending={}
        self.last_frame=None

    def update(self, frame, detections, masks):
        if self.last_frame is not None and frame != self.last_frame+1:
            self.pending.clear()
        self.last_frame=frame
        boxes=lambda rows:[[r[k] for k in ("x1","y1","x2","y2")] for r in rows]
        ov=self.iou(boxes(masks),boxes(detections))
        suggestions={}
        if ov.size:
            ri,ci=linear_sum_assignment(1-ov)
            for i,j in zip(ri,ci):
                others=np.r_[np.delete(ov[i],j),np.delete(ov[:,j],i)]
                if ov[i,j]>=.5 and ov[i,j]-(others.max() if len(others) else 0)>=.15:
                    suggestions[detections[j]["id"]]=self.seed_ids[int(masks[i]["id"])-1000]
        pending={}
        for tid,target in suggestions.items():
            old,count=self.pending.get(tid,(None,0))
            pending[tid]=(target,count+1 if old==target else 1)
            if pending[tid][1]>=self.persistence:
                self.alias[tid]=target
        self.pending=pending
        # Persistent aliases are proposals, not permission for duplicate outputs.
        # Prefer currently confirmed sources, then unchanged IDs, then stale aliases.
        order=sorted(detections,key=lambda r:(
            self.pending.get(r["id"],(None,0))[1]<self.persistence,
            self.alias.get(r["id"],r["id"])!=r["id"],-r["confidence"]))
        used=set(); output=[]
        for r in order:
            tid=r["id"]; target=self.alias.get(tid,tid)
            conflict=target in used
            if conflict:
                target=100000+tid
            used.add(target)
            output.append({**r,"id":target,"input_id":tid,
                           "association_state":"CONFLICT_UNKNOWN" if conflict else (
                               "SAM_CONFIRMED" if tid in suggestions else "PERSISTED_BASELINE_OR_ALIAS")})
        return output
