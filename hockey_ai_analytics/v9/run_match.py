"""Frozen external validation. Predictions never read future identity labels."""
from pathlib import Path
import sys,json,csv,argparse,time,hashlib,subprocess,collections
import cv2,numpy as np,torch,imageio_ffmpeg
from ultralytics import YOLO
from ultralytics.models.sam import SAM2VideoPredictor
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
sys.path[:0]=[str(BASE/'v7'),str(BASE/'v8')]
from benchmark import ious,score
from team_analysis import feature
from consensus import ConsensusGate

def write_csv(path,rows):
 with path.open('w') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def evaluate_selected(predictions,gt,target):
 rows=[]
 for f,box in enumerate(predictions,1):
  g=gt[(gt[:,0]==f)&(gt[:,1]==target),2:6].copy();g[:,2:]+=g[:,:2];state='lost';iou=None;other_match=None
  if box is not None:
   if len(g):iou=float(ious(g,[box])[0,0]);state='correct' if iou>=.5 else 'wrong_or_misaligned'
   else:state='output_while_target_absent'
   # Separate localization disagreement from a box matching another annotated identity.
   other=gt[(gt[:,0]==f)&(gt[:,1]!=target)];ob=other[:,2:6].copy();ob[:,2:]+=ob[:,:2]
   overlaps=ious(ob,[box]).reshape(-1)
   if len(overlaps) and overlaps.max()>=.5:other_match=int(other[overlaps.argmax(),1])
  rows.append(dict(frame=f,target_visible=bool(len(g)),state=state,iou=iou,other_gt_identity=other_match,box=json.dumps(box)))
 m={k:sum(r['state']==k for r in rows) for k in ['correct','lost','wrong_or_misaligned','output_while_target_absent']};m['visible_frames']=sum(r['target_visible'] for r in rows);m['visible_lost']=sum(r['target_visible'] and r['state']=='lost' for r in rows);m['matches_other_identity']=sum(r['other_gt_identity'] is not None and r['state']!='correct' for r in rows);m['correct_visible_coverage']=m['correct']/m['visible_frames'] if m['visible_frames'] else None
 return rows,m

def main():
 ap=argparse.ArgumentParser();ap.add_argument('sequence');args=ap.parse_args();seq=args.sequence;data=ROOT/'data'/seq;out=ROOT/'results'/seq;out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
 selection=json.loads((ROOT/'selections.json').read_text())[seq];seed=np.array(selection['bbox_xyxy'],float);target=selection['published_id']
 first=cv2.imread(str(data/'img1/000001.jpg'));H,W=first.shape[:2];w=848;h=round(H*w/W);sx=W/w;sy=H/h
 detector=YOLO(str(BASE/'v7/data/HockeyAI.pt'));tracks=[];features={};train=[];by=collections.defaultdict(list);start=time.time()
 for f in range(1,301):
  im=cv2.resize(cv2.imread(str(data/'img1'/f'{f:06d}.jpg')),(w,h));p=detector.track(im,persist=True,classes=[3,4,6],conf=.1,iou=.7,imgsz=960,tracker=str(BASE/'v5/botsort_reid.yaml'),verbose=False,device='cpu')[0]
  if p.boxes.id is not None:
   for box,tid,conf,cls in zip(p.boxes.xyxy.cpu().numpy(),p.boxes.id.int().tolist(),p.boxes.conf.tolist(),p.boxes.cls.int().tolist()):
    native=box*np.array([sx,sy,sx,sy]);r=dict(frame=f,id=tid,x1=float(native[0]),y1=float(native[1]),x2=float(native[2]),y2=float(native[3]),confidence=conf,cls=cls);tracks.append(r);by[f].append(r);v=feature(im,box);features[(f,tid)]=v
    if f<=30 and cls==4 and v is not None:train.append(v)
  if f%100==0:print(seq,'detector',f,flush=True)
 write_csv(out/'tracks.csv',tracks);detector_s=time.time()-start
 cv2.setRNGSeed(42);_,_,centers=cv2.kmeans(np.array(train),2,None,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,100,.001),10,cv2.KMEANS_PP_CENTERS);light=int(np.argmax(centers[:,0]+centers[:,2]-centers[:,1]));votes={};teams=[];team_by=collections.defaultdict(list)
 for r in tracks:
  v=features[(r['frame'],r['id'])];label='UNKNOWN';margin=0
  if r['cls']==6:label='REFEREE'
  elif v is not None:
   d=np.linalg.norm(centers-v,axis=1);votes[r['id']]=votes.get(r['id'],np.zeros(2))*.8+np.exp(-4*d);rank=np.argsort(votes[r['id']]);margin=float((votes[r['id']][rank[-1]]-votes[r['id']][rank[-2]])/max(votes[r['id']].sum(),1e-9))
   if margin>=.15:label='LIGHT' if rank[-1]==light else 'DARK'
  item={**r,'team_hypothesis':label,'team_margin':margin};teams.append(item);team_by[r['frame']].append(item)
 write_csv(out/'team_tracks.csv',teams)
 ov=ious([seed],[[r[k] for k in ['x1','y1','x2','y2']] for r in team_by[1]])[0];bound=team_by[1][int(ov.argmax())] if len(ov) and ov.max()>=.5 else None;seedteam=bound['team_hypothesis'] if bound else 'UNKNOWN';boundid=bound['id'] if bound else None
 source=out/'input.mp4';ff=imageio_ffmpeg.get_ffmpeg_exe()
 subprocess.run([ff,'-y','-framerate','30','-start_number','1','-i',str(data/'img1/%06d.jpg'),'-frames:v','300','-vf',f'scale={w}:{h},pad={w}:{h+h%2}:0:0','-c:v','libx264','-crf','0','-pix_fmt','yuv420p',str(source)],check=True,capture_output=True)
 del detector
 predictor=SAM2VideoPredictor(overrides=dict(model=str(BASE/'v8/models/sam2.1_t.pt'),imgsz=1024,device='cpu',save=False,verbose=False,conf=.25));sam=[];hybrid=[];fixed=[];gate=ConsensusGate(seedteam);start=time.time()
 for f,p in enumerate(predictor(source=str(source),bboxes=[(seed/np.array([sx,sy,sx,sy])).tolist()],stream=True),1):
  boxes=p.boxes.xyxy.cpu().numpy() if p.boxes is not None else np.empty((0,4));box=None
  if len(boxes)==1 and boxes[0,2]>boxes[0,0] and boxes[0,3]>boxes[0,1]:box=(boxes[0]*np.array([sx,sy,sx,sy])).tolist()
  sam.append(box);hybrid.append(gate.update(box,team_by[f]));candidate=next((r for r in by[f] if r['id']==boundid),None);fixed.append([candidate[k] for k in ['x1','y1','x2','y2']] if candidate else None)
  if f%50==0:print(seq,'SAM',f,round(time.time()-start,1),flush=True)
  if f in [1,150,300]:cv2.imwrite(str(out/f'sam_preview_{f}.jpg'),p.plot())
 # Evaluation is deliberately after inference, using separately loaded future reference rows.
 gt=np.loadtxt(data/'gt/gt.txt',delimiter=',');metrics={'sequence':seq,'team_scene':score(tracks,gt),'seed_team':seedteam,'initial_model_id':boundid,'selected_published_id':target,'detector_seconds':detector_s,'sam_seconds':time.time()-start,'selection':selection}
 for name,boxes in [('fixed_id',fixed),('sam',sam),('hybrid',hybrid)]:
  rows,m=evaluate_selected(boxes,gt,target);write_csv(out/f'{name}.csv',rows);metrics[name]=m
 (out/'metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2),flush=True)
if __name__=='__main__':main()
