from pathlib import Path
import csv,json,argparse
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--seed-only',action='store_true');a=ap.parse_args();out=ROOT/'ocr';out.mkdir(exist_ok=True);items=[]
selections=json.loads((ROOT/'selections.json').read_text())
for seq in json.loads((ROOT/'protocol.json').read_text())['sequences']:
 sel=selections[seq]
 if a.seed_only:rows=[dict(frame=1,box=json.dumps(sel['bbox_xyxy']))]
 else:rows=[r for r in csv.DictReader((ROOT/'results'/seq/'sam.csv').open()) if (int(r['frame'])-1)%30==0]
 for r in rows:
  b=json.loads(r['box'])
  if b is None:continue
  f=int(r['frame']);im=cv2.imread(str(ROOT/'data'/seq/'img1'/f'{f:06d}.jpg'));x1,y1,x2,y2=b;w=x2-x1;h=y2-y1
  crop=im[max(0,int(y1+.10*h)):min(im.shape[0],int(y1+.65*h)),max(0,int(x1+.1*w)):min(im.shape[1],int(x2-.1*w))]
  if crop.size==0:continue
  path=out/f'{seq}_{f:06d}.png';cv2.imwrite(str(path),cv2.resize(crop,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC));items.append(dict(sequence=seq,frame=f,path=str(path.resolve()),source='manual_initial_crop' if a.seed_only else 'SAM_hypothesis_crop'))
name='seed_inputs.json' if a.seed_only else 'inputs.json';(out/name).write_text(json.dumps(items,indent=2));print(len(items),name)
