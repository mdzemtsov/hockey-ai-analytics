from pathlib import Path
import csv,json
import numpy as np
from benchmark import ious
r=Path(__file__).resolve().parent;g=np.loadtxt(r/'data/personnel_gt.txt',delimiter=',');tracks=list(csv.DictReader((r/'results/hockey/tracks.csv').open()));selected=list(csv.DictReader((r/'results/hockey/selected.csv').open()));candidates=list(csv.DictReader((r/'results/teams/reappearance_candidates.csv').open()));counts=dict(eligible_visible_lost_frames=0,top1_correct=0,top3_contains_target=0,no_proposals=0)
for s in selected:
 f=int(s['frame'])
 if s['target_visible']!='True' or s['state']!='lost':continue
 counts['eligible_visible_lost_frames']+=1;gt=g[(g[:,0]==f)&(g[:,1]==1),2:6].copy();gt[:,2:]+=gt[:,:2];p=[c for c in candidates if int(c['frame'])==f];counts['no_proposals']+=not bool(p)
 good=[]
 for c in p:
  t=next(t for t in tracks if int(t['frame'])==f and t['id']==c['candidate_id']);ov=ious(gt,[[float(t[k]) for k in ['x1','y1','x2','y2']]])[0,0]
  if ov>=.5:good.append(int(c['rank']))
 counts['top1_correct']+=1 in good;counts['top3_contains_target']+=bool(good)
(r/'results/teams/recovery_metrics.json').write_text(json.dumps(counts,indent=2));print(counts)
