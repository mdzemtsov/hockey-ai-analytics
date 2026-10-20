"""Frozen V10-C on the 30-second holdout. No parameter fitting."""
from pathlib import Path
import json,csv,sys,time,collections,subprocess
import torch,numpy as np,imageio_ffmpeg
from ultralytics.models.sam import SAM2VideoPredictor
from stable_fusion import StableFusion
from evaluation import evaluate
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"v7"))
from benchmark import ious


def main():
    cfg=json.loads((ROOT/"long_holdout_protocol.json").read_text())
    seq=cfg["sequence"];n=cfg["last_frame"];data=ROOT/"data"/seq
    out=ROOT/"long_candidate"/seq;out.mkdir(parents=True,exist_ok=True)
    by=collections.defaultdict(list)
    with (ROOT/"long_baseline"/seq/"tracks.csv").open() as f:
        for r in csv.DictReader(f):
            r={k:float(v) for k,v in r.items()}
            for k in ("frame","id","cls"):r[k]=int(r[k])
            by[r["frame"]].append(r)
    assert (ROOT/"long_baseline"/seq/"metrics.json").exists()
    seeds=[r for r in by[1] if r["confidence"]>=.25]
    (out/"seeds.json").write_text(json.dumps(seeds,indent=2))
    (out/"frozen_candidate.json").write_text((ROOT/"stable_protocol.json").read_text())
    source=out/"input.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-y","-framerate",str(cfg["native_fps"]),
                    "-start_number","1","-i",str(data/"img1/%06d.jpg"),"-frames:v",str(n),
                    "-vf","scale=848:477,pad=848:478:0:0","-c:v","libx264","-crf","0",
                    "-pix_fmt","yuv420p",str(source)],check=True,capture_output=True)
    scale=np.array([1280/848,720/477]*2)
    prompts=[(np.array([r[k] for k in ("x1","y1","x2","y2")])/scale).tolist() for r in seeds]
    torch.set_num_threads(4)
    predictor=SAM2VideoPredictor(overrides=dict(model=str(ROOT.parent/"v8/models/sam2.1_t.pt"),
                                              imgsz=1024,device="cpu",save=False,verbose=False,conf=.25))
    fusion=StableFusion([r["id"] for r in seeds],ious)
    fields=["frame","id","x1","y1","x2","y2","confidence","cls"]
    rows=[];raw=[];start=time.time()
    with (out/"sam_tracks.csv").open("w") as sf,(out/"tracks.csv").open("w") as pf:
        sw=csv.DictWriter(sf,fieldnames=fields);sw.writeheader()
        pw=csv.DictWriter(pf,fieldnames=fields+["input_id","association_state"]);pw.writeheader()
        for frame,p in enumerate(predictor(source=str(source),bboxes=prompts,stream=True),1):
            current=[]
            for b,idx in zip(p.boxes.xyxy.cpu().numpy(),p.boxes.cls.int().tolist()):
                b=b*scale
                if b[2]<=b[0] or b[3]<=b[1]:continue
                assert 0<=idx<len(seeds)
                current.append(dict(frame=frame,id=1000+idx,x1=float(b[0]),y1=float(b[1]),
                                    x2=float(b[2]),y2=float(b[3]),confidence=1.,cls=seeds[idx]["cls"]))
            result=fusion.update(frame,by[frame],current)
            raw.extend(current);rows.extend(result)
            sw.writerows(current);pw.writerows(result)
            if frame%25==0:
                sf.flush();pf.flush()
                (out/"progress.json").write_text(json.dumps(dict(frame=frame,total=n,seconds=time.time()-start)))
                print(frame,round(time.time()-start,1),flush=True)
    assert frame==n
    # Evaluation starts only after inference and CSV completion.
    gt=np.loadtxt(data/"gt/gt.txt",delimiter=",")
    metrics=dict(candidate=evaluate(rows,gt,n),sam=evaluate(raw,gt,n),
                 baseline=json.loads((ROOT/"long_baseline"/seq/"metrics.json").read_text()),
                 seconds=time.time()-start,seed_count=len(seeds))
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics),flush=True)


if __name__=="__main__":main()
