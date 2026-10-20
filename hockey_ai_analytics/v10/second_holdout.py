from pathlib import Path
import json,csv,time
import cv2,numpy as np,torch
from ultralytics import YOLO
from evaluation import evaluate
ROOT=Path(__file__).resolve().parent


def main():
    cfg=json.loads((ROOT/"second_holdout_protocol.json").read_text())
    assert (ROOT/"second_holdout_protocol_sources.json").exists(),"Complete download required"
    seq=cfg["sequence"];n=cfg["last_frame"];torch.set_num_threads(4)
    for backend in cfg["backends"]:
        out=ROOT/"second_holdout"/backend/seq;out.mkdir(parents=True,exist_ok=True)
        if (out/"metrics.json").exists():continue
        config=ROOT.parent/"v5/botsort_reid.yaml" if backend=="baseline" else ROOT/"tracker_configs"/f"{backend}.yaml"
        model=YOLO(str(ROOT.parent/"v7/data/HockeyAI.pt"));rows=[];start=time.time()
        for frame in range(1,n+1):
            im=cv2.imread(str(ROOT/"data"/seq/"img1"/f"{frame:06d}.jpg"))
            assert im is not None
            p=model.track(cv2.resize(im,(848,477)),persist=True,classes=[3,4,6],conf=.1,
                          iou=.7,imgsz=960,tracker=str(config),verbose=False,device="cpu")[0]
            if p.boxes.id is not None:
                for box,tid,conf,cls in zip(p.boxes.xyxy.cpu().numpy(),p.boxes.id.int().tolist(),p.boxes.conf.tolist(),p.boxes.cls.int().tolist()):
                    b=box*np.array([1280/848,720/477]*2)
                    rows.append(dict(frame=frame,id=tid,x1=float(b[0]),y1=float(b[1]),x2=float(b[2]),y2=float(b[3]),confidence=conf,cls=cls))
            if frame%100==0:print(backend,seq,frame,flush=True)
        with (out/"tracks.csv").open("w") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        gt=np.loadtxt(ROOT/"data"/seq/"gt/gt.txt",delimiter=",")
        m=evaluate(rows,gt,n);m["seconds"]=time.time()-start
        (out/"metrics.json").write_text(json.dumps(m,indent=2));print(backend,json.dumps(m),flush=True)
        del model


if __name__=="__main__":main()
