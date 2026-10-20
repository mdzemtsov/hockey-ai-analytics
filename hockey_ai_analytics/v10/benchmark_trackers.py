from pathlib import Path
import sys,json,csv,time
import cv2,numpy as np,torch
from ultralytics import YOLO
from evaluation import evaluate
ROOT=Path(__file__).resolve().parent


def main():
    cfg=json.loads((ROOT/"tracker_protocol.json").read_text())
    torch.set_num_threads(4)
    for backend in cfg["backends"]:
        for seq in cfg["sequences"]:
            out=ROOT/"trackers"/backend/seq;out.mkdir(parents=True,exist_ok=True)
            if (out/"metrics.json").exists():continue
            model=YOLO(str(ROOT.parent/"v7/data/HockeyAI.pt"))
            rows=[];start=time.time()
            for frame in range(1,cfg["frames"]+1):
                im=cv2.imread(str(ROOT.parent/"v9/data"/seq/"img1"/f"{frame:06d}.jpg"))
                p=model.track(cv2.resize(im,(848,477)),persist=True,classes=[3,4,6],conf=.1,
                              iou=.7,imgsz=960,tracker=str(ROOT/"tracker_configs"/f"{backend}.yaml"),
                              verbose=False,device="cpu")[0]
                if p.boxes.id is not None:
                    for box,tid,conf,cls in zip(p.boxes.xyxy.cpu().numpy(),p.boxes.id.int().tolist(),p.boxes.conf.tolist(),p.boxes.cls.int().tolist()):
                        b=box*np.array([1280/848,720/477]*2)
                        rows.append(dict(frame=frame,id=tid,x1=float(b[0]),y1=float(b[1]),x2=float(b[2]),y2=float(b[3]),confidence=conf,cls=cls))
                if frame%100==0:print(backend,seq,frame,flush=True)
            with (out/"tracks.csv").open("w") as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            gt=np.loadtxt(ROOT.parent/"v9/data"/seq/"gt/gt.txt",delimiter=",")
            m=evaluate(rows,gt,cfg["frames"]);m["seconds"]=time.time()-start
            (out/"metrics.json").write_text(json.dumps(m,indent=2));print(backend,seq,json.dumps(m),flush=True)
            del model


if __name__=="__main__":main()
