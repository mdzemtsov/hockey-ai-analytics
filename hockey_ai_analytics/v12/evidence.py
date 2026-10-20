"""Image-only team and contrastive visual evidence; not verified personal identity."""
from pathlib import Path
import sys
import json
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "v11"))
sys.path.insert(0, str(ROOT.parent / "v10"))
from recovery import Recovery, iou, distance
from appearance_tracker import descriptor


def team_descriptor(image, box):
    x1,y1,x2,y2 = box
    w,h = x2-x1,y2-y1
    # Restrict to central upper jersey; exclude pants and much of the ice.
    a,b = np.clip([x1+.25*w,x1+.75*w],0,image.shape[1]).astype(int)
    c,d = np.clip([y1+.18*h,y1+.43*h],0,image.shape[0]).astype(int)
    crop = image[c:d,a:b]
    if crop.size < 60:
        return None
    pixels = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV).reshape(-1,3).astype(int)
    hue,sat,val = pixels.T
    bins=np.where(sat<55,24+np.minimum(val//32,7),np.minimum(hue//15,11)+12*(val>=128))
    hist=np.bincount(bins,minlength=32).astype(float)
    return hist/hist.sum()


class Encoder:
    def __init__(self):
        torch.set_num_threads(4)
        meta=json.loads((ROOT/"encoder_source.json").read_text())
        torch.hub.set_dir(str(ROOT/"hub"))
        repo=ROOT/"hub"/f"facebookresearch_dinov2_{meta['commit']}"
        self.model=torch.hub.load(str(repo),"dinov2_vits14",source="local",pretrained=True).eval()

    def encode(self,image,boxes):
        patches=[];valid=[]
        for box in boxes:
            x1,y1,x2,y2=box
            # Full person silhouette with little surrounding ice.
            x1,x2=np.clip([x1,x2],0,image.shape[1]).astype(int)
            y1,y2=np.clip([y1,y2],0,image.shape[0]).astype(int)
            crop=image[y1:y2,x1:x2]
            valid.append(crop.size>=60)
            crop=cv2.resize(crop,(112,224)) if crop.size else np.zeros((224,112,3),np.uint8)
            patches.append(crop[:,:,::-1].copy())
        if not patches:
            return []
        tensor=torch.from_numpy(np.stack(patches)).permute(0,3,1,2).float()/255
        tensor=(tensor-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
        with torch.inference_mode():
            emb=self.model(tensor)
            emb=torch.nn.functional.normalize(emb,dim=1).numpy()
        return [e if ok else None for e,ok in zip(emb,valid)]


class EvidenceGate:
    def __init__(self,config,teams,target_team,positive,negatives,identity=True):
        self.cfg=config;self.teams=teams;self.target=target_team
        self.positive=positive;self.negatives=negatives;self.identity=identity

    def check(self,candidate):
        if any(x is None for x in self.teams) or distance(*self.teams)<self.cfg["team_min_separation"]:
            return False,dict(reason="TEAM_CALIBRATION_UNCERTAIN",team="UNKNOWN")
        feat=candidate["team_feature"]
        ranks=sorted((distance(feat,t),i) for i,t in enumerate(self.teams))
        best,label=ranks[0];margin=ranks[1][0]-best
        team=label if best<=self.cfg["team_max_distance"] and margin>=self.cfg["team_margin"] else "UNKNOWN"
        info=dict(reason="TEAM_ACCEPTED",team=team,team_margin=margin)
        if team!=self.target:
            return False,{**info,"reason":"TEAM_UNKNOWN_OR_MISMATCH"}
        if not self.identity:
            return True,info
        emb=candidate["embedding"]
        if emb is None or self.positive is None:
            return False,{**info,"reason":"IDENTITY_UNAVAILABLE"}
        positive=float(emb@self.positive)
        negative=max((float(emb@n) for n in self.negatives),default=-1.)
        info.update(identity_similarity=positive,negative_similarity=negative,identity_margin=positive-negative)
        accepted=positive>=self.cfg["identity_min_cosine"] and positive-negative>=self.cfg["identity_negative_margin"]
        return accepted,{**info,"reason":"IDENTITY_ACCEPTED" if accepted else "IDENTITY_AMBIGUOUS"}


class GuardedRecovery:
    def __init__(self,base,gate):
        self.base=base;self.gate=gate

    def update(self,candidates):
        filtered=[];audit=[]
        for row in candidates:
            accepted,info=self.gate.check(row)
            # Retain all boxes for camera-motion and occlusion checks. Only disable
            # compatibility, avoiding an apparent absence of competing players.
            filtered.append({**row,"feature":row["feature"] if accepted else None})
            audit.append(dict(candidate_id=row["id"],accepted=accepted,**info))
        result=self.base.update(filtered)
        # First-frame manual selection is authoritative; automatic gates begin next frame.
        return {**result,"evidence":audit}
