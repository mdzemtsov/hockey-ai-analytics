from pathlib import Path
import csv,json,collections
import cv2,numpy as np
from team_calibration import TeamCalibration
ROOT=Path(__file__).resolve().parent


def main():
    cfg=json.loads((ROOT/"team_protocol.json").read_text());seq=cfg["sequence"]
    out=ROOT/"teams"/seq;out.mkdir(parents=True,exist_ok=True)
    first=cv2.imread(str(ROOT.parent/"v9/data"/seq/"img1/000001.jpg"))
    model=TeamCalibration(**{k:cfg[k] for k in ("min_separation","max_distance","min_margin")})
    for label,box in cfg["examples"].items():model.add(label,first,box)
    by=collections.defaultdict(list)
    with (ROOT.parent/"v9/results"/seq/"tracks.csv").open() as f:
        for r in csv.DictReader(f):by[int(r["frame"])].append(r)
    rows=[];tiles=[]
    for frame in range(1,301):
        im=cv2.imread(str(ROOT.parent/"v9/data"/seq/"img1"/f"{frame:06d}.jpg"))
        for r in by[frame]:
            box=[float(r[k]) for k in ("x1","y1","x2","y2")]
            result=model.classify(im,box,int(r["cls"]))
            rows.append(dict(frame=frame,input_id=int(r["id"]),**result))
            if frame in [1,60,120,180,240,300] and int(r["cls"])==4:
                x1,y1,x2,y2=np.array(box,int)
                crop=im[max(0,y1):min(im.shape[0],y2),max(0,x1):min(im.shape[1],x2)]
                if not crop.size:continue
                tile=np.zeros((180,160,3),np.uint8)
                h,w=crop.shape[:2];scale=min(145/w,140/h)
                thumb=cv2.resize(crop,(max(1,round(w*scale)),max(1,round(h*scale))))
                tile[:thumb.shape[0],:thumb.shape[1]]=thumb
                cv2.putText(tile,f"f{frame} id{r['id']}",(3,155),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                cv2.putText(tile,result["team"],(3,174),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
                tiles.append(tile)
    with (out/"hypotheses.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for start in range(0,len(tiles),24):
        page=tiles[start:start+24]
        while len(page)<24:page.append(np.zeros((180,160,3),np.uint8))
        sheet=np.vstack([np.hstack(page[i:i+6]) for i in range(0,24,6)])
        cv2.imwrite(str(out/f"review_{start//24}.jpg"),sheet)
    summary=dict(readiness=model.readiness(),counts=dict(collections.Counter(r["team"] for r in rows)),
                 warning="Coverage of hypotheses only, not accuracy; team ground truth unavailable.")
    (out/"summary.json").write_text(json.dumps(summary,indent=2));print(summary)


if __name__=="__main__":main()
