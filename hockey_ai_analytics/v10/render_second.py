from pathlib import Path
import json,subprocess
import cv2,numpy as np,imageio_ffmpeg
from review_long import read
ROOT=Path(__file__).resolve().parent


def main():
    seq="CNTRL_VS_PAC_001";out=ROOT/"second_review"/seq;out.mkdir(parents=True,exist_ok=True)
    methods=["baseline","ocsort","deepocsort"]
    bys={m:read(ROOT/"second_holdout"/m/seq/"tracks.csv") for m in methods}
    selected=json.loads((ROOT/"second_selection_summary.json").read_text())
    proc=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),"-y","-f","rawvideo","-vcodec","rawvideo",
                           "-pix_fmt","bgr24","-s","1696x956","-r","30","-i","-","-an",
                           "-c:v","libx264","-crf","22","-pix_fmt","yuv420p",str(out/"comparison.mp4")],
                          stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    for frame in range(1,901):
        im=cv2.imread(str(ROOT/"data"/seq/"img1"/f"{frame:06d}.jpg"));base=cv2.resize(im,(848,477))
        panels=[]
        for name in ["source"]+methods:
            panel=base.copy()
            if name!="source":
                dark=selected["DARK_TEST_08"][name]["initial_model_id"]
                light=selected["LIGHT_TEST_55"][name]["initial_model_id"]
                for r in bys[name][frame]:
                    b=(np.array([r[k] for k in ("x1","y1","x2","y2")])*np.array([848/1280,477/720]*2)).astype(int)
                    col=(0,255,255) if r["id"]==dark else ((255,80,255) if r["id"]==light else (60,190,60))
                    cv2.rectangle(panel,tuple(b[:2]),tuple(b[2:]),col,2 if r["id"] in (dark,light) else 1)
                    cv2.putText(panel,str(r["id"]),tuple(b[:2]-[0,4]),cv2.FONT_HERSHEY_SIMPLEX,.45,col,1)
            cv2.rectangle(panel,(0,0),(848,30),(20,20,20),-1)
            cv2.putText(panel,f"{name} | frame {frame} | {(frame-1)/30:.2f}s | experimental",(8,21),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            panels.append(cv2.copyMakeBorder(panel,0,1,0,0,cv2.BORDER_CONSTANT))
        joined=np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])])
        proc.stdin.write(joined.tobytes())
        if frame in (1,180,450,900):cv2.imwrite(str(out/f"review_{frame:04d}.jpg"),joined)
    proc.stdin.close();assert proc.wait()==0
    cap=cv2.VideoCapture(str(out/"comparison.mp4"));fps=cap.get(cv2.CAP_PROP_FPS);count=0
    while True:
        ok,_=cap.read()
        if not ok:break
        count+=1
    cap.release();assert count==900 and abs(fps-30)<.01
    (out/"video_validation.json").write_text(json.dumps(dict(decoded_frames=count,fps=fps,duration=count/fps),indent=2))
    print("Validated",count,"frames at",fps,"fps",flush=True)


if __name__=="__main__":main()
