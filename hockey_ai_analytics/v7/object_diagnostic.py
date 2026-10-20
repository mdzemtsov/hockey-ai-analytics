"""In-domain diagnostic only: model training overlap is unknown, not a holdout benchmark."""
from pathlib import Path
import zipfile,hashlib,json,csv
import cv2,numpy as np,torch
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from benchmark import ious
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(4);model=YOLO(str(ROOT/'data/HockeyAI.pt'));z=zipfile.ZipFile(ROOT/'data/HockeyAI_Dataset.zip')
names=sorted([n for n in z.namelist() if n.startswith('SHL/annotations/') and n.endswith('.txt')],key=lambda n:hashlib.sha256(n.encode()).hexdigest())[:40]
out=ROOT/'results/objects';out.mkdir(parents=True,exist_ok=True)
(out/'sample_manifest.json').write_text(json.dumps(names,indent=2));summaries={}
for width in [848,1920]:
 counts={i:dict(gt=0,predictions=0,tp=0,center_errors=[]) for i in range(7)};rows=[]
 for num,name in enumerate(names):
  im=cv2.imdecode(np.frombuffer(z.read(name.replace('/annotations/','/frames/').replace('.txt','.jpg')),np.uint8),1);h,w=im.shape[:2]
  im=cv2.resize(im,(width,round(h*width/w)));hh,ww=im.shape[:2]
  g=np.array([list(map(float,l.split())) for l in z.read(name).decode().splitlines() if l.strip()]);gb=g[:,1:5].copy();gb[:,:2]-=gb[:,2:]/2;gb[:,2:]+=gb[:,:2];gb*=np.array([ww,hh,ww,hh])
  pred=model.predict(im,conf=.25,imgsz=960 if width==848 else 1280,verbose=False,device='cpu')[0]
  pb=pred.boxes.xyxy.cpu().numpy();pc=pred.boxes.cls.int().cpu().numpy()
  for cls in range(7):
   a=gb[g[:,0]==cls];b=pb[pc==cls];c=counts[cls];c['gt']+=len(a);c['predictions']+=len(b)
   ov=ious(a,b);ri,ci=linear_sum_assignment(1-ov);ok=ov[ri,ci]>=.5;c['tp']+=int(ok.sum())
   for r,j in zip(ri[ok],ci[ok]):c['center_errors'].append(float(np.linalg.norm((a[r,:2]+a[r,2:]-b[j,:2]-b[j,2:])/2)))
  for box,cls,conf in zip(pb,pc,pred.boxes.conf.tolist()):rows.append(dict(sample=name,width=width,cls=int(cls),confidence=conf,x1=float(box[0]),y1=float(box[1]),x2=float(box[2]),y2=float(box[3])))
  if num==0:
   cv2.imwrite(str(out/f'preview_{width}.jpg'),pred.plot())
 summary={}
 for cls,c in counts.items():
  errs=c.pop('center_errors');summary[model.names[cls]]={**c,'precision':c['tp']/c['predictions'] if c['predictions'] else None,'recall':c['tp']/c['gt'] if c['gt'] else None,'matched_center_error_median_px':float(np.median(errs)) if errs else None}
 summaries[str(width)]=summary
 with (out/f'detections_{width}.csv').open('w') as h:wr=csv.DictWriter(h,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 print(width,summary,flush=True)
(out/'metrics.json').write_text(json.dumps(summaries,indent=2))
