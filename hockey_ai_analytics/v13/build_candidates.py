from pathlib import Path
import json,cv2,numpy as np,sys
R=Path(__file__).resolve().parent
P=json.loads((R/'protocol.json').read_text())
def iou(a,b):
 wh=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]));i=wh.prod()
 return i/max(1,np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-i)
gallery="--gallery" in sys.argv
records=[]
for seq in ([P['test_match']] if gallery else P['training_matches']):
 gt=np.loadtxt(R.parent/'v9/data'/seq/'gt/gt.txt',delimiter=',');by={}
 for f in range(1,121 if gallery else 301,5):
  rows=gt[gt[:,0]==f];boxes=[np.array([x,y,x+w,y+h]) for x,y,w,h in rows[:,2:6]]
  for row,b in zip(rows,boxes):
   if b[3]-b[1]<60 or b[2]-b[0]<20 or b[0]<2 or b[1]<2 or b[2]>1278 or b[3]>718:continue
   if any(j[1]!=row[1] and iou(b,bb)>.15 for j,bb in zip(rows,boxes)):continue
   by.setdefault(int(row[1]),[]).append(dict(sequence=seq,frame=f,id=int(row[1]),box=b.tolist()))
 selected=sorted([(k,v) for k,v in by.items() if len(v)>=(4 if gallery else 8)],key=lambda kv:(-len(kv[1]),kv[0]))[:16]
 out=R/'audit'/('gallery_'+seq if gallery else seq);out.mkdir(parents=True,exist_ok=True)
 lines=[]
 for identity,rs in selected:
  records+=rs;tiles=[]
  for index in np.linspace(0,len(rs)-1,4).astype(int):
   row=rs[index];im=cv2.imread(str(R.parent/'v9/data'/seq/'img1'/f"{row['frame']:06d}.jpg"));x1,y1,x2,y2=map(int,row['box'])
   crop=cv2.resize(im[y1:y2,x1:x2],(128,160));crop=cv2.copyMakeBorder(crop,30,0,0,0,cv2.BORDER_CONSTANT)
   cv2.putText(crop,f"id{identity} f{row['frame']}",(3,20),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1);tiles.append(crop)
  lines.append(cv2.hconcat(tiles))
 for start in range(0,len(lines),8):cv2.imwrite(str(out/f'sheet_{start//8}.jpg'),cv2.vconcat(lines[start:start+8]))
 print(seq,[(k,len(v)) for k,v in selected])
(R/('gallery_candidates.json' if gallery else 'candidate_crops.json')).write_text(json.dumps(records,indent=2))
