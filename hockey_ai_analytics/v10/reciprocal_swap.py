"""Only reversible reciprocal swaps; no global identity rewriting."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from appearance_tracker import descriptor,distance


class ReciprocalSwap:
    def __init__(self,seeds,first_image,iou_function):
        self.seeds=seeds;self.iou=iou_function;self.alias={};self.pending={}
        self.anchors={r["id"]:descriptor(first_image,[r[k] for k in ("x1","y1","x2","y2")]) for r in seeds}
        self.last_frame=None

    def update(self,frame,image,detections,masks):
        if self.last_frame is not None and frame!=self.last_frame+1:self.pending.clear()
        self.last_frame=frame
        boxes=lambda rows:[[r[k] for k in ("x1","y1","x2","y2")] for r in rows]
        ov=self.iou(boxes(masks),boxes(detections));suggestions={}
        if ov.size:
            ri,ci=linear_sum_assignment(1-ov)
            for i,j in zip(ri,ci):
                others=np.r_[np.delete(ov[i],j),np.delete(ov[:,j],i)]
                if ov[i,j]>=.5 and ov[i,j]-(others.max() if len(others) else 0)>=.15:
                    suggestions[detections[j]["id"]]=self.seeds[int(masks[i]["id"])-1000]["id"]
        current={r["id"]:r for r in detections}
        owner={self.alias.get(tid,tid):tid for tid in current}
        pending={};changed=set()
        for tid,desired in suggestions.items():
            original=self.alias.get(tid,tid)
            partner=owner.get(desired)
            if partner is None or partner==tid or suggestions.get(partner)!=original:continue
            pair=tuple(sorted([tid,partner]))
            if pair in pending:continue
            a,b=self.anchors.get(original),self.anchors.get(desired)
            if a is None or b is None or distance(a,b)<.25:continue
            f1=descriptor(image,boxes([current[tid]])[0])
            f2=descriptor(image,boxes([current[partner]])[0])
            if f1 is None or f2 is None:continue
            if not (distance(f1,b)<=.5 and distance(f2,a)<=.5 and
                    distance(f1,a)-distance(f1,b)>=.15 and distance(f2,b)-distance(f2,a)>=.15):continue
            # Include directed assignment so opposite requests cannot accumulate.
            request=tuple(sorted([(tid,desired),(partner,original)]))
            previous,count=self.pending.get(pair,(None,0))
            pending[pair]=(request,count+1 if request==previous else 1)
            if pending[pair][1]>=3:
                self.alias[tid]=desired;self.alias[partner]=original
                changed.update(pair)
        self.pending=pending
        output=[{**r,"id":self.alias.get(r["id"],r["id"]),"input_id":r["id"],
                 "association_state":"RECIPROCAL_SWAP" if r["id"] in changed else
                 ("PERSISTED_SWAP" if self.alias.get(r["id"],r["id"])!=r["id"] else "BASELINE")}
                for r in detections]
        assert len({r["id"] for r in output})==len(output), "Permutation must preserve unique IDs"
        return output
