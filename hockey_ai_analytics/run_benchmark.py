import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from make_review import ROOT,load_video
from tracker_v4 import run,CONFIG


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def digitize_v3(frames,pts):
    rows=[];previous=np.array([195.,295.])
    for i,frame in enumerate(frames):
        row=dict(frame_index=i,pts_s=pts[i],state='NOT_STARTED' if i<90 else 'LOST',x='',y='',w='',h='',observed_id='',reason='DIGITIZED_FROM_RENDER_NOT_ORIGINAL_LOG')
        if i>=90:
            hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            mask=(((hsv[:,:,0]>=20)&(hsv[:,:,0]<=40))&(hsv[:,:,1]>180)&(hsv[:,:,2]>180)).astype('uint8')*255
            mask[:75]=0
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            candidates=[]
            for c in contours:
                x,y,w,h=cv2.boundingRect(c)
                if 25<h<110 and 15<w<75 and h>w and len(c)>20:
                    center=np.array([x+(w-1)/2,y+(h-1)/2])
                    distance=np.linalg.norm(center-previous)
                    if distance<50:candidates.append((distance,x,y,w,h,center))
            if candidates:
                _,x,y,w,h,center=min(candidates,key=lambda t:t[0]);previous=center
                # Keep marker geometry, explicitly NOT a player bbox.
                row.update(state='TRACKING',x=x,y=y,w=w,h=h)
        rows.append(row)
    return rows


def main():
    frames,pts=load_video(ROOT/'evidence/source.mov')
    v3,v3pts=load_video(ROOT/'evidence/red_track_v3_conservative.mp4')
    if len(v3)!=len(frames):raise ValueError('Frame counts differ; cannot align by index')
    # Frame-order alignment: compare untouched image region in each frame.
    maes=[float(np.mean(np.abs(a[80:250,450:800].astype(float)-b[80:250,450:800].astype(float)))) for a,b in zip(frames,v3)]
    result=ROOT/'results';result.mkdir(exist_ok=True)
    pred3=digitize_v3(v3,pts);pred4=run(frames,pts)
    write_csv(result/'v3_digitized.csv',pred3);write_csv(result/'v4_predictions_raw.csv',pred4)
    (result/'v4_config.json').write_text(json.dumps(CONFIG,indent=2)+'\n')
    manifest={'source_frames':len(frames),'source_pts_first':pts[:3],'source_pts_last':pts[-3:],
              'v3_pts_first':v3pts[:3],'v3_pts_last':v3pts[-3:],
              'source_v3_unmarked_region_mae_mean':float(np.mean(maes)),
              'source_v3_unmarked_region_mae_max':float(max(maes)),
              'v3_alignment':'frame_index; measured PTS retained separately',
              'files':{}}
    for path in [ROOT/'evidence/source.mov',ROOT/'evidence/red_track_v3_conservative.mp4',ROOT/'manual_anchors.csv',ROOT/'tracker_v4.py']:
        manifest['files'][str(path.relative_to(ROOT))]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (result/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name,rows in [('V3',pred3),('V4',pred4)]:
        from collections import Counter
        print(name,Counter(r['state'] for r in rows))
        print('first_failure',next((r for r in rows[90:] if r['state']!='TRACKING'),None))


if __name__=='__main__':main()
