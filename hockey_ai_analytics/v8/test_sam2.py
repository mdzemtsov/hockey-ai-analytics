from pathlib import Path
import sys,time,json,csv,subprocess,hashlib
import cv2,numpy as np,torch,imageio_ffmpeg
from ultralytics.models.sam import SAM2VideoPredictor
ROOT=Path(__file__).resolve().parent;V7=ROOT.parent/'v7';sys.path.insert(0,str(V7));from benchmark import ious
out=ROOT/'results/sam2';out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
source=out/'input.mp4';ff=imageio_ffmpeg.get_ffmpeg_exe()
if not source.exists():subprocess.run([ff,'-y','-framerate','30','-start_number','1','-i',str(V7/'data/frames/%06d.jpg'),'-frames:v','300','-vf','scale=848:477,pad=848:478:0:0','-c:v','libx264','-crf','0','-pix_fmt','yuv420p',str(source)],check=True,capture_output=True)
weights=ROOT/'models/sam2.1_t.pt';weights.parent.mkdir(exist_ok=True)
predictor=SAM2VideoPredictor(overrides=dict(model=str(weights),imgsz=1024,device='cpu',save=False,verbose=False,conf=.25))
gt=np.loadtxt(V7/'data/personnel_gt.txt',delimiter=',');seed=gt[(gt[:,0]==1)&(gt[:,1]==1),2:6].copy();seed[:,2:]+=seed[:,:2];seed*=np.array([848/1280,477/720]*2)
rows=[];start=time.time()
for f,pred in enumerate(predictor(source=str(source),bboxes=seed.tolist(),stream=True),1):
 b=pred.boxes.xyxy.cpu().numpy() if pred.boxes is not None else np.empty((0,4));b=b*np.array([1280/848,720/477]*2)
 target=gt[(gt[:,0]==f)&(gt[:,1]==1),2:6].copy();target[:,2:]+=target[:,:2]
 state='lost';ov=None;box=b[0].tolist() if len(b)==1 and b[0,2]>b[0,0] and b[0,3]>b[0,1] else None
 if box is not None:
  if len(target):ov=float(ious(target,[box])[0,0]);state='correct' if ov>=.5 else 'wrong_or_misaligned'
  else:state='output_while_target_absent'
 rows.append(dict(frame=f,target_visible=bool(len(target)),state=state,iou=ov,box=json.dumps(box)))
 if f%30==0:print('SAM2',f,round(time.time()-start,1),flush=True)
 if f in [1,60,180,240,300]:cv2.imwrite(str(out/f'review_{f}.jpg'),pred.plot())
with (out/'selected.csv').open('w') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
metrics={s:sum(r['state']==s for r in rows) for s in ['correct','lost','wrong_or_misaligned','output_while_target_absent']};metrics.update(frames=len(rows),visible_frames=sum(r['target_visible'] for r in rows),elapsed_s=time.time()-start,weight_sha256=hashlib.sha256(weights.read_bytes()).hexdigest())
(out/'metrics.json').write_text(json.dumps(metrics,indent=2));print(metrics)
