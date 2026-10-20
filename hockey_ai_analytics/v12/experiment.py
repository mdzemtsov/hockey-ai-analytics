"""Three-way ablation with saved predictions before evaluation labels are read."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import time
import cv2
import numpy as np
from evidence import Encoder, Recovery, EvidenceGate, GuardedRecovery, descriptor, team_descriptor, iou

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"v11"))
from run_recovery import read_tracks, score


def csv_write(path,rows):
    with path.open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("spec")
    args=parser.parse_args();spec=json.loads(Path(args.spec).read_text())
    out=ROOT/spec["output"];out.mkdir(parents=True,exist_ok=True)
    images=ROOT.parent/spec["images"];track_path=ROOT.parent/spec["tracks"]
    tracks=read_tracks(track_path);first,last=spec["first"],spec["last"]
    cfg=json.loads((ROOT/"config.json").read_text())
    basecfg=json.loads((ROOT.parent/"v11/recovery_config.json").read_text())
    encoder=Encoder()
    image=cv2.imread(str(images/f"{first:06d}.jpg"))
    boxes=[s["bbox_xyxy"] for s in spec["targets"]]
    teams=[team_descriptor(image,b) for b in boxes]
    seeds=encoder.encode(image,boxes)
    initial=tracks[first]
    initial_emb=encoder.encode(image,[r["box"] for r in initial])
    engines={};outputs={}
    for index,seed in enumerate(spec["targets"]):
        negatives=[e for r,e in zip(initial,initial_emb) if e is not None and iou(np.array(seed["bbox_xyxy"]),np.array(r["box"]))<cfg["negative_exclusion_iou"]]
        for variant in ("v11","team_only","team_identity"):
            base=Recovery(basecfg,seed["bbox_xyxy"],descriptor(image,seed["bbox_xyxy"]),spec["fps"])
            engine=base if variant=="v11" else GuardedRecovery(base,EvidenceGate(cfg,teams,index,seeds[index],negatives,variant=="team_identity"))
            key=(seed["name"],variant);engines[key]=engine;outputs[key]=[]
    start=time.time();cache=out/"features.npz"
    # Cache is keyed to source files, crop order and model; no labels are cached.
    signature=hashlib.sha256(track_path.read_bytes()+(ROOT/"encoder_source.json").read_bytes()+
                            (ROOT/"evidence.py").read_bytes())
    for frame in range(first,last+1):
        signature.update((images/f"{frame:06d}.jpg").read_bytes())
    signature=signature.hexdigest()
    saved=np.load(cache) if cache.exists() else None
    if saved is not None:
        assert str(saved["signature"])==signature,"Stale feature cache"
    all_emb=[];offset=0
    for frame in range(first,last+1):
        image=cv2.imread(str(images/f"{frame:06d}.jpg"));assert image is not None
        rows=tracks[frame]
        embs=saved["embeddings"][offset:offset+len(rows)] if saved is not None else encoder.encode(image,[r["box"] for r in rows])
        offset+=len(rows)
        candidates=[]
        for row,emb in zip(rows,embs):
            if emb is not None and np.isnan(emb).any():emb=None
            all_emb.append(emb if emb is not None else np.full(384,np.nan))
            candidates.append({**row,"feature":descriptor(image,row["box"]),
                               "team_feature":team_descriptor(image,row["box"]),"embedding":emb})
        for key,engine in engines.items():
            outputs[key].append(dict(source_frame=frame,**engine.update(candidates)))
        if (frame-first+1)%100==0:print(spec["output"],frame-first+1,f"{time.time()-start:.1f}s",flush=True)
    if saved is None:np.savez_compressed(cache,signature=signature,embeddings=np.array(all_emb))
    for (name,variant),predictions in outputs.items():
        folder=out/name/variant;folder.mkdir(parents=True,exist_ok=True)
        (folder/"predictions.json").write_text(json.dumps(predictions,indent=2))
    # Only now load future reference.
    gt_path=ROOT.parent/spec["gt"];gt=np.loadtxt(gt_path,delimiter=",")
    summary=[]
    for seed in spec["targets"]:
        folder=out/seed["name"]
        request=dict(first_source_frame=first,last_source_frame=last,fps=spec["fps"],seed_bbox_xyxy=seed["bbox_xyxy"])
        (folder/"request.json").write_text(json.dumps(request,indent=2))
        ranks=sorted([(iou(np.array(seed["bbox_xyxy"]),np.array([r[2],r[3],r[2]+r[4],r[3]+r[5]])),int(r[1])) for r in gt[gt[:,0]==first]],reverse=True)
        assert ranks and ranks[0][0]>=.5 and (len(ranks)==1 or ranks[0][0]-ranks[1][0]>=.1),"Ambiguous reference seed"
        target=ranks[0][1];refs=[];others=[]
        for frame in range(first,last+1):
            present=gt[gt[:,0]==frame];matched=present[present[:,1]==target]
            row=dict(frame=frame-first+1,source_frame=frame,target_id=target,visible=bool(len(matched)),x1="",y1="",x2="",y2="")
            if len(matched):
                x,y,w,h=matched[0,2:6];row.update(x1=x,y1=y,x2=x+w,y2=y+h)
            refs.append(row)
            for p in present:
                x,y,w,h=p[2:6];others.append(dict(frame=frame-first+1,id=int(p[1]),x1=x,y1=y,x2=x+w,y2=y+h))
        csv_write(folder/"reference.csv",refs);csv_write(folder/"all_reference.csv",others)
        for variant in ("v11","team_only","team_identity"):
            metrics=score(folder,outputs[(seed["name"],variant)],folder/variant)
            summary.append(dict(target=seed["name"],variant=variant,**metrics))
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    (out/"provenance.json").write_text(json.dumps(dict(input_signature=signature,gt_sha256=hashlib.sha256(gt_path.read_bytes()).hexdigest(),human_verified=False),indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
