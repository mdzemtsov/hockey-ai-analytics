from pathlib import Path
import json,csv,time,configparser
import cv2,numpy as np,torch
from ultralytics import YOLO
from evaluation import evaluate
ROOT=Path(__file__).resolve().parent


def main():
    cfg=json.loads((ROOT/"long_holdout_protocol.json").read_text())
    seq=cfg["sequence"]; n=cfg["last_frame"]
    data=ROOT/"data"/seq; out=ROOT/"long_baseline"/seq
    out.mkdir(parents=True,exist_ok=True)
    assert (ROOT/"long_holdout_sources.json").exists(), "Download manifest incomplete"
    ini=configparser.ConfigParser(); ini.read(data/"seqinfo.ini")
    assert ini.getint("Sequence","frameRate")==cfg["native_fps"]
    torch.set_num_threads(4)
    model=YOLO(str(ROOT.parent/"v7/data/HockeyAI.pt"))
    rows=[]; started=time.time()
    with (out/"tracks.csv").open("w") as f:
        fields=["frame","id","x1","y1","x2","y2","confidence","cls"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for frame in range(1,n+1):
            im=cv2.imread(str(data/"img1"/f"{frame:06d}.jpg"))
            assert im is not None
            H,W=im.shape[:2]; h=round(H*848/W)
            p=model.track(cv2.resize(im,(848,h)),persist=True,classes=[3,4,6],conf=.1,
                          iou=.7,imgsz=960,tracker=str(ROOT.parent/"v5/botsort_reid.yaml"),
                          verbose=False,device="cpu")[0]
            if p.boxes.id is not None:
                for box,tid,conf,cls in zip(p.boxes.xyxy.cpu().numpy(),p.boxes.id.int().tolist(),
                                           p.boxes.conf.tolist(),p.boxes.cls.int().tolist()):
                    b=box*np.array([W/848,H/h]*2)
                    r=dict(frame=frame,id=tid,x1=float(b[0]),y1=float(b[1]),x2=float(b[2]),
                           y2=float(b[3]),confidence=conf,cls=cls)
                    rows.append(r);writer.writerow(r)
            if frame%100==0:
                f.flush()
                print(frame,round(time.time()-started,1),flush=True)
    gt=np.loadtxt(data/"gt/gt.txt",delimiter=",")
    metrics=evaluate(rows,gt,n)
    metrics.update(sequence=seq,native_fps=cfg["native_fps"],seconds=time.time()-started)
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics),flush=True)


if __name__=="__main__":
    main()
