from pathlib import Path
import sys,json,zipfile,time,csv
import numpy as np,cv2,torch
from scipy.optimize import linear_sum_assignment
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
ROOT=Path(__file__).resolve().parent;V7=ROOT.parent/'v7';sys.path.insert(0,str(V7));from benchmark import ious
torch.set_num_threads(4)
m=AutoDetectionModel.from_pretrained(model_type='ultralytics',model_path=str(V7/'data/HockeyAI.pt'),confidence_threshold=.25,device='cpu',image_size=960)
z=zipfile.ZipFile(V7/'data/HockeyAI_Dataset.zip');names=json.loads((V7/'results/objects/sample_manifest.json').read_text());out=ROOT/'results/sahi';out.mkdir(parents=True,exist_ok=True)
counts=dict(gt=0,predictions=0,tp=0);rows=[];start=time.time()
for n,name in enumerate(names):
 im=cv2.imdecode(np.frombuffer(z.read(name.replace('/annotations/','/frames/').replace('.txt','.jpg')),np.uint8),1);h,w=im.shape[:2];im=cv2.resize(im,(848,round(h*848/w)));h,w=im.shape[:2]
 gt=np.array([list(map(float,l.split())) for l in z.read(name).decode().splitlines() if l.strip()]);b=gt[gt[:,0]==5,1:5].copy();b[:,:2]-=b[:,2:]/2;b[:,2:]+=b[:,:2];b*=np.array([w,h,w,h])
 pred=get_sliced_prediction(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),m,slice_height=384,slice_width=384,overlap_height_ratio=.2,overlap_width_ratio=.2,perform_standard_pred=True,postprocess_type='NMS',postprocess_match_metric='IOU',postprocess_match_threshold=.5,verbose=0)
 p=[v for v in pred.object_prediction_list if v.category.id==5];pb=[v.bbox.to_xyxy() for v in p];ov=ious(b,pb);ri,ci=linear_sum_assignment(1-ov);tp=int((ov[ri,ci]>=.5).sum());counts['gt']+=len(b);counts['predictions']+=len(p);counts['tp']+=tp
 rows.append(dict(sample=name,gt=len(b),predictions=len(p),tp=tp))
 for v in p:
  x1,y1,x2,y2=map(int,v.bbox.to_xyxy());cv2.rectangle(im,(x1,y1),(x2,y2),(0,0,255),1)
 if n<3:cv2.imwrite(str(out/f'preview_{n}.jpg'),im)
 if (n+1)%10==0:print(n+1,counts,flush=True)
counts.update(precision=counts['tp']/counts['predictions'] if counts['predictions'] else 0,recall=counts['tp']/counts['gt'],elapsed_s=time.time()-start)
(out/'metrics.json').write_text(json.dumps(counts,indent=2))
with (out/'per_image.csv').open('w') as f:wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
print(counts)
