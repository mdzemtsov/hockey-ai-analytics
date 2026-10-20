"""Build reference-only QA artifacts and seed-only inference manifests."""
from pathlib import Path
import json,csv,hashlib,subprocess
import cv2,numpy as np,imageio_ffmpeg
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent


def write_csv(path,rows):
    with path.open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    protocol=json.loads((ROOT/"protocol.json").read_text());seq=protocol["development_source"]
    data=BASE/"v9/data"/seq
    gt=np.loadtxt(data/"gt/gt.txt",delimiter=",")
    manifest=[];review=[]
    for case in protocol["cases"]:
        out=ROOT/"cases"/case["id"];out.mkdir(parents=True,exist_ok=True)
        seed=gt[(gt[:,0]==case["first"])&(gt[:,1]==case["target"])]
        assert len(seed)==1,"Seed absent or ambiguous"
        x,y,w,h=seed[0,2:6]
        request=dict(case_id=case["id"],source_sequence=seq,first_source_frame=case["first"],
                     last_source_frame=case["last"],fps=protocol["fps"],seed_frame=1,
                     seed_bbox_xyxy=[x,y,x+w,y+h],seed_origin="publisher_first_frame_surrogate")
        # No target identity or future positions in inference request.
        (out/"request.json").write_text(json.dumps(request,indent=2))
        refs=[];all_refs=[];tiles=[]
        frame_list=list(range(case["first"],case["last"]+1))
        samples=set(np.linspace(case["first"],case["last"],12).astype(int).tolist()+[case["event"]])
        for f in frame_list:
            path=data/"img1"/f"{f:06d}.jpg";im=cv2.imread(str(path));assert im is not None
            manifest.append(dict(path=str(path.relative_to(BASE)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            matches=gt[(gt[:,0]==f)&(gt[:,1]==case["target"])]
            ref=dict(frame=f-case["first"]+1,source_frame=f,pts_s=(f-case["first"])/protocol["fps"],
                     target_id=case["target"],visible=bool(len(matches)),x1="",y1="",x2="",y2="",
                     reference_source="publisher",independent_human_verified=False)
            if len(matches):
                x,y,w,h=matches[0,2:6];ref.update(x1=x,y1=y,x2=x+w,y2=y+h)
            refs.append(ref)
            for g in gt[gt[:,0]==f]:
                x,y,w,h=g[2:6]
                all_refs.append(dict(frame=f-case["first"]+1,source_frame=f,id=int(g[1]),
                                     x1=x,y1=y,x2=x+w,y2=y+h))
            if f in samples:
                if len(matches):
                    x,y,w,h=matches[0,2:6].astype(int)
                    cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,255),3)
                cv2.putText(im,f"{case['id']} source {f} | publisher reference",(15,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
                tiles.append(cv2.resize(im,(426,240)))
        write_csv(out/"reference.csv",refs);write_csv(out/"all_reference.csv",all_refs)
        while len(tiles)%3:tiles.append(np.zeros((240,426,3),np.uint8))
        cv2.imwrite(str(out/"contact_sheet.jpg"),np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)]))
        # Unannotated original video is the inference input, not GT overlays.
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-y","-framerate",str(protocol["fps"]),
                        "-start_number",str(case["first"]),"-i",str(data/"img1/%06d.jpg"),
                        "-frames:v",str(len(frame_list)),"-c:v","libx264","-crf","0","-pix_fmt","yuv420p",
                        str(out/"source.mp4")],check=True,capture_output=True)
        cap=cv2.VideoCapture(str(out/"source.mp4"));count=0
        while True:
            ok,_=cap.read()
            if not ok:break
            count+=1
        cap.release();assert count==len(frame_list)
        review.append(dict(case_id=case["id"],group=case["group"],frames=count,
                           visible_frames=sum(r["visible"] for r in refs),
                           review_status="PUBLISHER_REFERENCE_PENDING_INDEPENDENT_REVIEW",
                           identity_ambiguity_notes="",reviewer="",verified=False))
    unique={r["path"]:r for r in manifest}
    (ROOT/"source_manifest.json").write_text(json.dumps(list(unique.values()),indent=2))
    write_csv(ROOT/"review_queue.csv",review)
    (ROOT/"reference_provenance.json").write_text(json.dumps(dict(
        source="https://github.com/harshap-ai/VIP-HTD",
        file=str((data/"gt/gt.txt").relative_to(BASE)),
        sha256=hashlib.sha256((data/"gt/gt.txt").read_bytes()).hexdigest(),
        annotations_modified=False,interpolated_frames=0,human_verified_in_this_project=False),indent=2))
    print(json.dumps(dict(cases=len(review),unique_source_frames=len(unique),case_frames=sum(r["frames"] for r in review))))


if __name__=="__main__":main()
