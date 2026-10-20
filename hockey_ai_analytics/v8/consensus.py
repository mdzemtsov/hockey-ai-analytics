"""Conservative experimental agreement gate; no ground-truth access."""
import numpy as np

def overlap(a,b):
 lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:]);inter=np.maximum(hi-lo,0).prod();aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return float(inter/max(aa+bb-inter,1e-9))
class ConsensusGate:
 def __init__(self,team):self.team=team;self.last=None;self.stable=0
 def update(self,sam_box,detections):
  ranked=[]
  if sam_box and self.team in ('LIGHT','DARK'):
   for d in detections:
    if int(d['cls'])!=4 or d['team_hypothesis']!=self.team:continue
    b=[float(d[k]) for k in ['x1','y1','x2','y2']];ranked.append((overlap(sam_box,b),d['id'],b))
  ranked.sort(key=lambda x:x[0],reverse=True)
  if not ranked or ranked[0][0]<.5 or (len(ranked)>1 and ranked[0][0]-ranked[1][0]<.15):self.last=None;self.stable=0;return None
  _,ident,box=ranked[0];self.stable=self.stable+1 if ident==self.last else 1;self.last=ident
  return box if self.stable>=3 else None
