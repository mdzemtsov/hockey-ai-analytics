import json,sys,cv2,numpy as np
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent/'v6/work/jobs'
jobs={'original':'53c675251de742cca5c1d1c5e0a2e1f1','chicago':'26f0582c1ee54ad4aa807304e5183a6c','carolina':'a5c5d9912aa14f18bcf0ce5d6fb13d73'}
rows=[]
for name,jid in jobs.items():
 frames=json.loads((B/jid/'observations.json').read_text())
 for split,m in [('development',frames[0]),('validation',frames[-1])]:
  im=cv2.imread(str(B/jid/f"frame_{m['frame_index']}.jpg"))
  for n,o in enumerate(m['objects']):
   if o['kind']!='player':continue
   x,y,w,h=o['bbox'];crop=im[max(0,int(y)):int(y+h),max(0,int(x)):int(x+w)]
   ident=len(rows);p=R/f'crop_{ident}.jpg';cv2.imwrite(str(p),crop)
   rows.append(dict(id=ident,clip=name,split=split,job=jid,frame=m['frame_index'],object_index=n,box=o['bbox'],baseline=o['color'],crop=p.name))
for split in ['development','validation']:
 subset=[r for r in rows if r['split']==split];sheet=np.full((((len(subset)+7)//8)*180,8*140,3),235,np.uint8)
 for i,r in enumerate(subset):
  im=cv2.imread(str(R/r['crop']));scale=min(120/im.shape[1],140/im.shape[0]);im=cv2.resize(im,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST);x=(i%8)*140;y=(i//8)*180
  sheet[y+25:y+25+im.shape[0],x:x+im.shape[1]]=im;cv2.putText(sheet,f"{r['id']} {r['clip']}",(x+2,y+16),0,.4,(0,0,0),1)
 cv2.imwrite(str(R/f'{split}.jpg'),sheet)
(R/'manifest.json').write_text(json.dumps(rows,indent=2))
print(len(rows))
