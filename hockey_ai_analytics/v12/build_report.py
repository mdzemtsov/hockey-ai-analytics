"""Render three-way comparisons and summarize abstention, never alter predictions."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import cv2
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parent
VARIANTS=("v11","team_only","team_identity")


def write_csv(path,rows):
    with path.open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def render(folder,images,request):
    predictions={v:json.loads((folder/v/"predictions.json").read_text()) for v in VARIANTS}
    refs=list(csv.DictReader((folder/"reference.csv").open()))
    output=folder/"comparison.mp4"
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),"-y","-loglevel","error","-f","rawvideo",
         "-pix_fmt","bgr24","-s","1920x400","-r",str(request["fps"]),"-i","-",
         "-an","-c:v","libx264","-crf","20","-pix_fmt","yuv420p",str(output)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    for i,ref in enumerate(refs):
        frame=request["first_source_frame"]+i
        source=cv2.imread(str(images/f"{frame:06d}.jpg"));assert source is not None
        panels=[]
        for variant in VARIANTS:
            p=predictions[variant][i];im=source.copy()
            reference=[float(ref[k]) for k in ("x1","y1","x2","y2")] if ref["visible"]=="True" else None
            for box,color in [(reference,(255,0,255)),(p["box"],(0,220,0))]:
                if box is not None:
                    b=[round(x) for x in box];cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),color,3)
            im=cv2.resize(im,(640,360));im=cv2.copyMakeBorder(im,40,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(im,f"{variant} | {p['state']} | {frame}",(7,24),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            panels.append(im)
        proc.stdin.write(cv2.hconcat(panels).tobytes())
    proc.stdin.close();assert proc.wait()==0
    cap=cv2.VideoCapture(str(output));count=0
    while cap.read()[0]:count+=1
    cap.release();assert count==len(refs)
    for variant,rows in predictions.items():
        exported=[]
        for p in rows:
            b=p["box"]
            exported.append(dict(source_frame=p["source_frame"],state=p["state"],
                candidate_id=p.get("candidate_id"),x1=b[0] if b else "",y1=b[1] if b else "",
                x2=b[2] if b else "",y2=b[3] if b else ""))
        write_csv(folder/variant/"predictions.csv",exported)
    return dict(path=str(output.relative_to(ROOT)),decoded_frames=count)


def main():
    summaries=[];videos=[];diagnostics=[];sections=[]
    for stage in ("development","validation"):
        spec=json.loads((ROOT/f"{stage}.json").read_text())
        summary=json.loads((ROOT/stage/"summary.json").read_text())
        summaries += [dict(stage=stage,**row) for row in summary]
        sections.append(f"<h2>{stage}</h2>")
        for seed in spec["targets"]:
            name=seed["name"];folder=ROOT/stage/name
            request=json.loads((folder/"request.json").read_text())
            videos.append(render(folder,ROOT.parent/spec["images"],request))
            baseline=list(csv.DictReader((folder/"v11/evaluation.csv").open()))
            table=""
            for r in [r for r in summary if r["target"]==name]:
                table+=f"<tr><td>{r['variant']}</td><td>{r['correct']}/{r['visible_frames']}</td><td>{r['wrong_person_frames']}</td><td>{r['output_while_absent']}</td></tr>"
                if r["variant"]!="v11":
                    evaluated=list(csv.DictReader((folder/r["variant"]/"evaluation.csv").open()))
                    diagnostics.append(dict(stage=stage,target=name,variant=r["variant"],
                        baseline_correct_now_missed=sum(a["state"]=="CORRECT" and b["state"]=="VISIBLE_MISS" for a,b in zip(baseline,evaluated)),
                        baseline_wrong_person_now_abstained=sum(a["wrong_person"]=="True" and b["state"] in ("VISIBLE_MISS","ABSENT") for a,b in zip(baseline,evaluated))))
            sections.append(f"<h3>{name}</h3><table><tr><th>Variant</th><th>Correct / visible</th><th>Wrong person</th><th>Output while absent</th></tr>{table}</table>"
                            f'<video controls preload="metadata" src="{stage}/{name}/comparison.mp4"></video>')
    write_csv(ROOT/"summary.csv",summaries)
    (ROOT/"abstention_audit.json").write_text(json.dumps(diagnostics,indent=2))
    (ROOT/"video_validation.json").write_text(json.dumps(videos,indent=2))
    (ROOT/"review.html").write_text('<!doctype html><meta charset="utf-8"><title>V12 evidence comparison</title>'
        '<style>body{font:17px system-ui;max-width:1400px;margin:30px auto;padding:16px}video{width:100%}td,th{padding:10px;text-align:left}h2{margin-top:50px}</style>'
        '<h1>Team and identity evidence — experimental</h1>'
        '<p>Left: V11. Middle: team check. Right: team + DINOv2 identity comparison. '
        'Green is output; magenta is unchanged publisher reference. No independent human review.</p>'
        '<p>Zero wrong-player frames does not imply success: compare correct coverage and misses. '
        'The new gates are not deployed in the user application.</p>'+''.join(sections))
    files=[]
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or "hub" in p.parts or p.name=="artifact_manifest.json":continue
        files.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (ROOT/"artifact_manifest.json").write_text(json.dumps(files,indent=2))
    print(json.dumps(dict(videos=videos,artifacts=len(files)),indent=2))


if __name__=="__main__":main()
