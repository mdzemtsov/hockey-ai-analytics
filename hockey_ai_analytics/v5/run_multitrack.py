"""YOLO11s + BoT-SORT with detector features. Track IDs are hypotheses, not verified identities."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'runtime')
os.environ['MPLCONFIGDIR']=str(ROOT/'runtime/matplotlib')
os.environ['YOLO_AUTOINSTALL']='false'
import sys,json,time,csv,hashlib
sys.path.insert(0,str(ROOT.parent))
import cv2,numpy as np,torch
from ultralytics import YOLO,settings
from make_review import load_video
from detect_probe import in_ice,team
from run_benchmark import write_csv
from target_gate import TargetGate

SEED_FRAME=90
SEED_POINT=np.array([195.,302.])


def main():
    settings.update({k:False for k in ['sync','wandb','mlflow','comet'] if k in settings})
    torch.set_num_threads(4);torch.manual_seed(0);cv2.setRNGSeed(0);cv2.setNumThreads(1)
    frames,pts=load_video(ROOT.parent/'evidence/source.mov')
    model=YOLO(str(ROOT/'models/yolo11s.pt'))
    output=ROOT/'results';output.mkdir(exist_ok=True)
    all_rows=[];target_rows=[];gate=TargetGate(SEED_FRAME,SEED_POINT)
    start=time.monotonic()
    for i,frame in enumerate(frames):
        res=model.track(frame,persist=True,tracker=str(ROOT/'botsort_reid.yaml'),
            classes=[0],imgsz=1280,conf=.10,iou=.55,device='cpu',verbose=False)[0]
        objects=[]
        if res.boxes.id is not None:
            for box,confidence,tid in zip(res.boxes.xyxy.cpu().numpy(),res.boxes.conf.cpu().numpy(),res.boxes.id.cpu().numpy()):
                if not in_ice(box):continue
                label,fraction=team(frame,box)
                x,y,z,t=map(float,box)
                r=dict(frame_index=i,pts_s=pts[i],track_id=int(tid),x=x,y=y,w=z-x,h=t-y,
                    detector_conf=float(confidence),team_hypothesis=label,red_fraction=fraction,
                    identity_status='AUTOMATIC_UNVERIFIED',role='UNKNOWN_PLAYER_GOALIE_OR_OFFICIAL')
                objects.append(r);all_rows.append(r)
        target=gate.update(i,pts[i],objects)
        target_rows.append(target)
        if i%30==0:print(json.dumps({'frame':i,'objects':len(objects),'bound_id':gate.bound_id,'target_state':target['state'],'seconds':round(time.monotonic()-start,1)}),flush=True)
    write_csv(output/'all_tracks.csv',all_rows)
    write_csv(output/'target_v5_raw.csv',target_rows)
    metadata={'model':'YOLO11s COCO person','tracker':'BoT-SORT, with_reid=True, model=auto detector features',
        'imgsz':1280,'conf':.10,'iou':.55,'device':'cpu','seed_frame':SEED_FRAME,'seed_point':SEED_POINT.tolist(),
        'bound_model_track_id':gate.bound_id,'frames':len(frames),'inference_seconds':time.monotonic()-start,
        'source_sha256':hashlib.sha256((ROOT.parent/'evidence/source.mov').read_bytes()).hexdigest(),
        'weights_sha256':hashlib.sha256((ROOT/'models/yolo11s.pt').read_bytes()).hexdigest(),
        'no_ground_truth_read_by_tracking':True,'identity_claim':'UNVERIFIED_MODEL_IDS; manual audit required',
        'puck_tracking':False,'possession_inference':False,'tactical_inference':False}
    (output/'run_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    from collections import Counter
    print('DONE',dict(Counter(r['state'] for r in target_rows)),flush=True)


if __name__=='__main__':main()
