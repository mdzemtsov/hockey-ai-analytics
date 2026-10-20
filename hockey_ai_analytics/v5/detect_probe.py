"""Local-only COCO person detection probe. No video upload or player identity claim."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'runtime')
os.environ['MPLCONFIGDIR']=str(ROOT/'runtime/matplotlib')
os.environ['YOLO_AUTOINSTALL']='false'
import sys,json,time
sys.path.insert(0,str(ROOT.parent))
import cv2,numpy as np,torch
from PIL import Image,ImageDraw
from ultralytics import YOLO,settings
from make_review import load_video
settings.update({k:False for k in ['sync','wandb','mlflow','comet'] if k in settings})
torch.set_num_threads(4)
ICE=np.array([[0,178],[180,94],[590,72],[848,87],[848,464],[600,464],[120,330],[0,252]],np.int32)


def in_ice(box):
    x,y,z,t=box
    return cv2.pointPolygonTest(ICE,(float((x+z)/2),float(t)),False)>=0 and 12<t-y<120 and 4<z-x<95


def team(frame,box):
    x,y,z,t=map(int,box);w=z-x;h=t-y
    crop=frame[max(0,int(y+h*.2)):min(464,int(y+h*.68)),max(0,int(x+w*.2)):min(848,int(z-w*.2))]
    if crop.size==0:return 'UNKNOWN',0.
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    red=((hsv[:,:,0]<13)|(hsv[:,:,0]>165))&(hsv[:,:,1]>85)&(hsv[:,:,2]>40)
    fraction=float(red.mean())
    return ('RED' if fraction>.13 else 'NONRED_UNKNOWN_ROLE'),fraction


def main():
    frames,pts=load_video(ROOT.parent/'evidence/source.mov')
    model=YOLO(str(ROOT/'models/yolo11s.pt'))
    indices=[0,30,60,90,120,150,180,210,216,240,270,300,330,360,375,390,420,423]
    out=[];can=Image.new('RGB',(3*848,6*484),'white');d=ImageDraw.Draw(can)
    start=time.monotonic()
    for j,i in enumerate(indices):
        res=model.predict(frames[i],classes=[0],imgsz=1280,conf=.10,iou=.55,device='cpu',verbose=False)[0]
        det=[];f=frames[i].copy()
        for box,confidence in zip(res.boxes.xyxy.cpu().numpy(),res.boxes.conf.cpu().numpy()):
            if not in_ice(box):continue
            label,fraction=team(frames[i],box);det.append(dict(xyxy=box.tolist(),conf=float(confidence),team=label,red_fraction=fraction))
            x,y,z,t=map(int,box);col=(0,0,255) if label=='RED' else (255,100,0)
            cv2.rectangle(f,(x,y),(z,t),col,1);cv2.putText(f,f'{len(det)} {confidence:.2f}',(x,max(8,y-2)),cv2.FONT_HERSHEY_SIMPLEX,.3,col,1)
        out.append(dict(frame_index=i,pts_s=pts[i],detections=det))
        x=j%3*848;y=j//3*484;can.paste(Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)),(x,y));d.text((x+3,y+465),f'frame {i}; persons in ice ROI: {len(det)}',fill='black')
        print(i,len(det),flush=True)
    (ROOT/'probe_detections.json').write_text(json.dumps(out,indent=2)+'\n')
    can.save(ROOT/'probe_contact.jpg',quality=95)
    print('seconds',time.monotonic()-start,flush=True)


if __name__=='__main__':main()
