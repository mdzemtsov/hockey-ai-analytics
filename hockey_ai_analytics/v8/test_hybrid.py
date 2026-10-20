"""Experimental consensus. Rules fixed before SAM metrics; GT used only for scoring."""
from pathlib import Path
import sys,csv,json
import numpy as np
ROOT=Path(__file__).resolve().parent;V7=ROOT.parent/'v7';sys.path.insert(0,str(V7));from benchmark import ious
sam=list(csv.DictReader((ROOT/'results/sam2/selected.csv').open()));detections=list(csv.DictReader((V7/'results/teams/team_tracks.csv').open()));gt=np.loadtxt(V7/'data/personnel_gt.txt',delimiter=',')
seed=gt[(gt[:,0]==1)&(gt[:,1]==1),2:6].copy();seed[:,2:]+=seed[:,:2];first=[r for r in detections if int(r['frame'])==1];ov=ious(seed,[[float(r[k]) for k in ['x1','y1','x2','y2']] for r in first])[0];seed_team=first[int(ov.argmax())]['team_hypothesis']
from consensus import ConsensusGate
gate=ConsensusGate(seed_team);rows=[]
for r in sam:
 f=int(r['frame']);b=json.loads(r['box']);chosen=gate.update(b,[d for d in detections if int(d['frame'])==f])
 target=gt[(gt[:,0]==f)&(gt[:,1]==1),2:6].copy();target[:,2:]+=target[:,:2];state='lost';overlap=None
 if chosen:
  if len(target):overlap=float(ious(target,[chosen])[0,0]);state='correct' if overlap>=.5 else 'wrong_or_misaligned'
  else:state='output_while_target_absent'
 rows.append(dict(frame=f,target_visible=bool(len(target)),state=state,iou=overlap,box=json.dumps(chosen),identity_status='EXPERIMENTAL_HYPOTHESIS_NOT_HUMAN_CONFIRMED'))
out=ROOT/'results/hybrid';out.mkdir(exist_ok=True)
with (out/'selected.csv').open('w') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
metrics={s:sum(r['state']==s for r in rows) for s in ['correct','lost','wrong_or_misaligned','output_while_target_absent']};metrics['visible_frames']=sum(r['target_visible'] for r in rows);(out/'metrics.json').write_text(json.dumps(metrics,indent=2));print(metrics)
