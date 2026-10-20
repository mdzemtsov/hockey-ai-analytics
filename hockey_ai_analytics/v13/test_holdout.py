"""Query features use publisher boxes; labels enter retrieval scoring only."""
from pathlib import Path
import sys,json,hashlib
import cv2,numpy as np
from retrieval import project,rank,summarize,save_rows
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/"v12"))
from evidence import Encoder


def main():
    for entry in json.loads((R/"candidate_lock.json").read_text())["files"]:
        assert hashlib.sha256((R.parent/entry["path"]).read_bytes()).hexdigest()==entry["sha256"]
    p=json.loads((R/"protocol.json").read_text());seq=p["test_match"]
    gt=np.loadtxt(R.parent/"v9/data"/seq/"gt/gt.txt",delimiter=",")
    labels=json.loads((R/"gallery_labels.json").read_text())["teams"]
    rows=[];features=[];encoder=Encoder()
    for frame in range(p["test_query_frames"][0],p["test_query_frames"][1]+1,p["query_stride"]):
        image=cv2.imread(str(R/"test/img1"/f"{frame:06d}.jpg"));assert image is not None
        candidates=[];boxes=[]
        for row in gt[gt[:,0]==frame]:
            identity=int(row[1]);x,y,w,h=row[2:6]
            # Keep occluded queries; only reject crops too small or outside image.
            if h<60 or w<20 or x<2 or y<2 or x+w>1278 or y+h>718:continue
            boxes.append([x,y,x+w,y+h])
            candidates.append(dict(sequence=seq,frame=frame,id=identity,identity=f"{seq}:{identity}",
                team=f"{seq}:{labels.get(str(identity),'UNVERIFIED')}",box=boxes[-1]))
        emb=encoder.encode(image,boxes)
        for row,e in zip(candidates,emb):
            assert e is not None;rows.append(row);features.append(e)
    features=np.array(features)
    np.savez_compressed(R/"test/features.npz",features=features)
    (R/"test/queries.json").write_text(json.dumps(rows,indent=2))
    gallery=json.loads((R/"gallery.json").read_text());ge=np.load(R/"gallery.npz")["features"]
    calibration=json.loads((R/"calibration.json").read_text());summary={}
    for trained in (False,True):
        queries_emb=project(features,trained);gallery_emb=project(ge,trained)
        for views in (1,4):
            name=f"{'trained' if trained else 'generic'}_{views}"
            ranked=rank(gallery,gallery_emb,rows,queries_emb,views)
            threshold=calibration[name]["threshold"]
            for row in ranked:row["accepted"]=row["margin"]>=threshold
            save_rows(R/"test"/f"{name}.csv",ranked)
            summary[name]=summarize(ranked,threshold)
    (R/"test/summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
