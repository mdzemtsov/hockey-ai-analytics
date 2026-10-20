"""Fixed, external 300-frame benchmark. GT is used only by scoring and first-frame selection."""
from pathlib import Path
import json,csv,time,argparse
import cv2,numpy as np,torch
from ultralytics import YOLO
import motmetrics as mm
ROOT=Path(__file__).resolve().parent.parent/'v7'

def ious(a,b):
 a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
 lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:])
 inter=np.maximum(hi-lo,0).prod(2);union=(a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-inter
 return inter/np.maximum(union,1e-9)

def score(rows,gt):
 acc=mm.MOTAccumulator(auto_id=True)
 for f in range(1,301):
  g=gt[gt[:,0]==f];p=[r for r in rows if r['frame']==f]
  gb=g[:,2:6].copy();gb[:,2:]+=gb[:,:2]
  pb=[[r[k] for k in ['x1','y1','x2','y2']] for r in p]
  d=1-ious(gb,pb);d[d>0.5]=np.nan
  acc.update(g[:,1].astype(int).tolist(),[r['id'] for r in p],d)
 keys=['num_frames','num_objects','num_predictions','num_matches','num_misses','num_false_positives','num_switches','precision','recall','idf1','mota']
 return {k:float(v) for k,v in mm.metrics.create().compute(acc,metrics=keys).iloc[0].items()}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('variant',choices=['baseline','hockey']);args=ap.parse_args()
 torch.set_num_threads(4)
 out=ROOT.parent/'v8/results/high_detail';out.mkdir(parents=True,exist_ok=True)
 model=YOLO(str(ROOT.parent/'v5/models/yolo11s.pt' if args.variant=='baseline' else ROOT/'data/HockeyAI.pt'))
 classes=[0] if args.variant=='baseline' else [3,4,6]
 gt=np.loadtxt(ROOT/'data/personnel_gt.txt',delimiter=',');rows=[];selected=[];selected_id=None;start=time.time()
 writer=cv2.VideoWriter(str(out/'review.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(848,478))
 for f in range(1,301):
  im=cv2.imread(str(ROOT/'data/frames'/f'{f:06d}.jpg'));im=cv2.resize(im,(1280,720))
  result=model.track(im,persist=True,classes=classes,conf=.1,iou=.7,imgsz=1280,tracker=str(ROOT.parent/'v5/botsort_reid.yaml'),verbose=False,device='cpu')[0]
  current=[]
  if result.boxes.id is not None:
   for box,tid,conf,cls in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.id.int().tolist(),result.boxes.conf.tolist(),result.boxes.cls.int().tolist()):
    x1,y1,x2,y2=box
    r=dict(frame=f,id=tid,x1=float(x1),y1=float(y1),x2=float(x2),y2=float(y2),confidence=conf,cls=cls);rows.append(r);current.append(r)
  if f==1 and current:
   seed=gt[(gt[:,0]==1)&(gt[:,1]==1),2:6].copy();seed[:,2:]+=seed[:,:2]
   ov=ious(seed,[[r[k] for k in ['x1','y1','x2','y2']] for r in current])[0]
   if ov.max()>=.5:selected_id=current[int(ov.argmax())]['id']
  chosen=next((r for r in current if r['id']==selected_id),None)
  target=gt[(gt[:,0]==f)&(gt[:,1]==1)]
  state='lost';overlap=None
  if chosen:
   pb=[[chosen[k] for k in ['x1','y1','x2','y2']]]
   if len(target):
    b=target[:,2:6].copy();b[:,2:]+=b[:,:2];overlap=float(ious(b,pb)[0,0]);state='correct' if overlap>=.5 else 'wrong_or_misaligned'
   else:state='output_while_target_absent'
  selected.append(dict(frame=f,selected_id=selected_id,target_visible=bool(len(target)),state=state,iou=overlap))
  im=cv2.resize(im,(848,477))
  for r in current:
   a=np.array([r[k] for k in ['x1','y1','x2','y2']])*np.array([848/1280,477/720]*2);x1,y1,x2,y2=a.astype(int)
   color=(0,255,255) if r['id']==selected_id else (80,190,70)
   cv2.rectangle(im,(x1,y1),(x2,y2),color,1);cv2.putText(im,str(r['id']),(x1,y1-3),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
  cv2.putText(im,f'{args.variant} frame {f} | fixed selected ID {selected_id}',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
  writer.write(cv2.copyMakeBorder(im,0,1,0,0,cv2.BORDER_CONSTANT))
  if f%30==0:print(args.variant,f,round(time.time()-start,1),flush=True)
 writer.release()
 for name,data in [('tracks',rows),('selected',selected)]:
  with (out/f'{name}.csv').open('w') as h:w=csv.DictWriter(h,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 metrics=score(rows,gt);metrics['elapsed_seconds']=time.time()-start
 metrics['selected_player']={s:sum(r['state']==s for r in selected) for s in ['correct','lost','wrong_or_misaligned','output_while_target_absent']}
 metrics['selected_player']['visible_frames']=sum(r['target_visible'] for r in selected)
 (out/'metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
