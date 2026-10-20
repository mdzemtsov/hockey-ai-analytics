from pathlib import Path
import json,csv,time,hashlib
import cv2,numpy as np,torch
from ultralytics import YOLO
R=Path(__file__).resolve().parent
for x in json.loads((R/'candidate_lock.json').read_text())['files']:
 assert hashlib.sha256((R.parent/x['path']).read_bytes()).hexdigest()==x['sha256']
p=json.loads((R/'protocol.json').read_text())['validation'];out=R/'validation'
assert len(json.loads((out/'source_manifest.json').read_text()))==p['last']-p['first']+1
assert not (out/'tracks.csv').exists(),'Use saved tracks or explicitly remove to rerun'
torch.set_num_threads(4);model=YOLO(str(R.parent/'v7/data/HockeyAI.pt'));rows=[];start=time.time()
for frame in range(p['first'],p['last']+1):
 im=cv2.imread(str(out/'img1'/f'{frame:06d}.jpg'));assert im is not None
 pred=model.track(cv2.resize(im,(848,477)),persist=True,classes=[3,4,6],conf=.1,iou=.7,imgsz=960,tracker=str(R.parent/'v5/botsort_reid.yaml'),verbose=False,device='cpu')[0]
 if pred.boxes.id is not None:
  for box,tid,cls in zip(pred.boxes.xyxy.cpu().numpy(),pred.boxes.id.int().tolist(),pred.boxes.cls.int().tolist()):
   b=box*np.array([1280/848,720/477]*2);rows.append(dict(frame=frame,id=tid,cls=cls,x1=b[0],y1=b[1],x2=b[2],y2=b[3]))
 if (frame-p['first']+1)%100==0:print('detector',frame-p['first']+1,f'{time.time()-start:.1f}s',flush=True)
with (out/'tracks.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
