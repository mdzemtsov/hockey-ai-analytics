"""Prepare reference proposals from manual anchors; these are NOT certified GT."""
import csv
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parent


def load_video(path):
    cap = cv2.VideoCapture(str(path))
    frames, pts = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        pts.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
    cap.release()
    return frames, pts


def prepare():
    frames, pts = load_video(ROOT/'evidence/source.mov')
    anchors = list(csv.DictReader((ROOT/'manual_anchors.csv').open()))
    aa = np.array([[float(r[k]) for k in ['frame_index','cx','cy','w','h']] for r in anchors])
    vals = np.column_stack([np.interp(np.arange(len(frames)), aa[:,0], aa[:,j]) for j in range(1,5)])
    proposals = []
    for i,(cx,cy,w,h) in enumerate(vals):
        proposals.append(dict(frame_index=i,pts_s=pts[i],target_id='RED_TEST_01',status='visible',
            x=round(cx-w/2,2),y=round(cy-h/2,2),w=round(w,2),h=round(h,2),
            provenance='manual' if i in aa[:,0] else 'interpolated',
            reviewer='assistant_visual_anchors_NOT_HUMAN',notes='PROPOSAL_PENDING_FRAME_REVIEW'))
    with (ROOT/'reference_proposals.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(proposals[0]));wr.writeheader();wr.writerows(proposals)
    folder=ROOT/'evidence/review';folder.mkdir(exist_ok=True)
    for start in range(0,len(frames),48):
        can=Image.new('RGB',(1200,8*160),'#eeeeee');d=ImageDraw.Draw(can)
        for j,i in enumerate(range(start,min(start+48,len(frames)))):
            cx,cy,w,h=vals[i];left=int(cx)-50;top=int(cy)-35
            crop=frames[i][top:top+70,left:left+100].copy()
            cv2.rectangle(crop,(int(cx-w/2-left),int(cy-h/2-top)),(int(cx+w/2-left),int(cy+h/2-top)),(0,255,0),1)
            cv2.drawMarker(crop,(int(cx-left),int(cy-top)),(255,0,0),cv2.MARKER_CROSS,5,1)
            im=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)).resize((200,140))
            x=j%6*200;y=j//6*160;can.paste(im,(x,y));d.text((x+3,y+141),f'{i} ({cx:.0f},{cy:.0f})',fill='black')
        can.save(folder/f'frames_{start:03}.jpg',quality=95)
    print('prepared',len(frames),'frames; pts',pts[:3],pts[-3:])


if __name__=='__main__':prepare()
