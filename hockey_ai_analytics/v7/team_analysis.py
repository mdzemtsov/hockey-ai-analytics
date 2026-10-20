"""Two modes share detections. Jersey clusters are team hypotheses, never identity proof."""
from pathlib import Path
import csv,json,collections
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
from benchmark import ious
ROOT=Path(__file__).resolve().parent

def feature(im,b):
 x1,y1,x2,y2=b;w=x2-x1;h=y2-y1
 crop=im[max(0,int(y1+.18*h)):min(im.shape[0],int(y1+.53*h)),max(0,int(x1+.2*w)):min(im.shape[1],int(x2-.2*w))]
 if crop.size<12:return None
 hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV);v=hsv[:,:,2]/255;s=hsv[:,:,1]/255
 # White jersey: high brightness / low chroma; dark/coloured jersey: lower brightness or more chroma.
 return np.array([np.median(v),np.median(s),np.mean((v>.65)&(s<.25)),np.mean(s>.35)],np.float32)

def main():
 rows=list(csv.DictReader((ROOT/'results/hockey/tracks.csv').open()));by=collections.defaultdict(list)
 for r in rows:
  for k in ['frame','id','cls']:r[k]=int(r[k])
  for k in ['x1','y1','x2','y2','confidence']:r[k]=float(r[k])
  by[r['frame']].append(r)
 features={};train=[]
 for f in range(1,301):
  im=cv2.resize(cv2.imread(str(ROOT/'data/frames'/f'{f:06d}.jpg')),(848,477))
  for r in by[f]:
   b=np.array([r[k] for k in ['x1','y1','x2','y2']])*np.array([848/1280,477/720]*2);vec=feature(im,b);features[(f,r['id'])]=vec
   if f<=30 and r['cls']==4 and vec is not None:train.append(vec)
 cv2.setRNGSeed(42);_,_,centers=cv2.kmeans(np.array(train),2,None,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,100,.001),10,cv2.KMEANS_PP_CENTERS)
 light=int(np.argmax(centers[:,0]+centers[:,2]-centers[:,1]));votes={};out=[]
 for f in range(1,301):
  for r in by[f]:
   v=features[(f,r['id'])];label='UNKNOWN';margin=0
   if r['cls']==6:label='REFEREE'
   elif v is not None:
    d=np.linalg.norm(centers-v,axis=1);weights=np.exp(-4*d);old=votes.get(r['id'],np.zeros(2));votes[r['id']]=old*.8+weights;rank=np.argsort(votes[r['id']]);margin=float((votes[r['id']][rank[-1]]-votes[r['id']][rank[-2]])/max(votes[r['id']].sum(),1e-9))
    if margin>=.15:label='LIGHT' if rank[-1]==light else 'DARK'
   out.append({**r,'team_hypothesis':label,'team_margin':margin})
 truth={1:'LIGHT',2:'DARK',3:'DARK',4:'DARK',5:'DARK',6:'LIGHT',7:'LIGHT',8:'DARK',9:'DARK',10:'LIGHT',25:'LIGHT',26:'REFEREE',38:'REFEREE',39:'REFEREE',41:'REFEREE',43:'DARK',48:'REFEREE'}
 gt=np.loadtxt(ROOT/'data/personnel_gt.txt',delimiter=',');counts=collections.Counter();confusion=collections.Counter()
 for f in range(1,301):
  g=gt[gt[:,0]==f];p=[r for r in out if r['frame']==f];b=g[:,2:6].copy();b[:,2:]+=b[:,:2];ov=ious(b,[[r[k] for k in ['x1','y1','x2','y2']] for r in p]);ri,ci=linear_sum_assignment(1-ov)
  for i,j in zip(ri,ci):
   if ov[i,j]<.5:continue
   expected=truth[int(g[i,1])];actual=p[j]['team_hypothesis'];confusion[(expected,actual)]+=1;counts['matched']+=1
   if actual=='UNKNOWN':counts['abstained']+=1
   elif actual==expected:counts['correct']+=1
   else:counts['wrong']+=1
 # Rank reappearance suggestions using seed appearance only, never rebind automatically.
 seed_gt=gt[(gt[:,0]==1)&(gt[:,1]==1),2:6].copy();seed_gt[:,2:]+=seed_gt[:,:2]
 first=by[1];ov=ious(seed_gt,[[r[k] for k in ['x1','y1','x2','y2']] for r in first])[0];seed=first[int(ov.argmax())];seedvec=features[(1,seed['id'])];seed_team=next(r['team_hypothesis'] for r in out if r['frame']==1 and r['id']==seed['id']);proposals=[]
 for f in range(2,301):
  if any(r['id']==seed['id'] for r in by[f]):continue
  ranked=[]
  for r in out:
   if r['frame']!=f or r['cls']!=4 or (seed_team not in ('LIGHT','DARK') or r['team_hypothesis']!=seed_team):continue
   v=features[(f,r['id'])]
   if v is not None:ranked.append((float(np.linalg.norm(v-seedvec)),r))
  for rank,(dist,r) in enumerate(sorted(ranked,key=lambda a:a[0])[:3],1):proposals.append(dict(frame=f,rank=rank,candidate_id=r['id'],appearance_distance=dist,status='REQUIRES_MANUAL_IDENTITY_CONFIRMATION'))
 dest=ROOT/'results/teams';dest.mkdir(parents=True,exist_ok=True)
 for name,data in [('team_tracks',out),('reappearance_candidates',proposals)]:
  with (dest/f'{name}.csv').open('w') as h:w=csv.DictWriter(h,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 metrics=dict(counts);metrics['conditional_accuracy']=counts['correct']/(counts['correct']+counts['wrong']);metrics['classification_coverage']=1-counts['abstained']/counts['matched'];metrics['confusion']={f'{a}->{b}':v for (a,b),v in confusion.items()}
 (dest/'metrics.json').write_text(json.dumps(metrics,indent=2));(dest/'visual_reference.json').write_text(json.dumps(dict(provenance='Assistant visual jersey/role review of source crops, not independent human validation',identity_to_team=truth,fit_frames=[1,30],centers=centers.tolist()),indent=2));print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
