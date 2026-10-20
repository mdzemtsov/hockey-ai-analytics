"""Post-inference QA only. Native-rate video and temporal diagnostic metrics."""
from pathlib import Path
import json,csv,sys,collections,subprocess
import cv2,numpy as np,imageio_ffmpeg
from evaluation import evaluate
ROOT=Path(__file__).resolve().parent


def read(path):
    by=collections.defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            for k in ("frame","id","cls"):r[k]=int(r[k])
            for k in ("x1","y1","x2","y2","confidence"):r[k]=float(r[k])
            by[r["frame"]].append(r)
    return by


def main():
    seq="CAR_VS_NYR_001";out=ROOT/"long_review"/seq;out.mkdir(parents=True,exist_ok=True)
    bys={name:read(ROOT/folder/seq/"tracks.csv") for name,folder in
         [("Baseline","long_baseline"),("Rejected V10-C","long_candidate")]}
    gt=np.loadtxt(ROOT/"data"/seq/"gt/gt.txt",delimiter=",")
    windows=[]
    for start in range(1,1801,300):
        g=gt[(gt[:,0]>=start)&(gt[:,0]<start+300)].copy();g[:,0]-=start-1
        row=dict(first_frame=start,last_frame=start+299,start_seconds=(start-1)/60)
        for name,by in bys.items():
            pred=[{**r,"frame":r["frame"]-start+1} for f in range(start,start+300) for r in by[f]]
            row[name]=evaluate(pred,g,300)
        windows.append(row)
    (out/"windows.json").write_text(json.dumps(windows,indent=2))
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    proc=subprocess.Popen([ff,"-y","-f","rawvideo","-vcodec","rawvideo","-pix_fmt","bgr24",
                           "-s","1696x478","-r","60","-i","-","-an","-c:v","libx264",
                           "-crf","21","-pix_fmt","yuv420p",str(out/"comparison.mp4")],
                          stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    for frame in range(1,1801):
        im=cv2.imread(str(ROOT/"data"/seq/"img1"/f"{frame:06d}.jpg"))
        panels=[]
        for name,by in bys.items():
            panel=cv2.resize(im,(848,477))
            for r in by[frame]:
                b=(np.array([r[k] for k in ("x1","y1","x2","y2")])*np.array([848/1280,477/720]*2)).astype(int)
                color=(0,180,255) if r.get("association_state")=="CONFLICT_UNKNOWN" else (90,220,90)
                cv2.rectangle(panel,tuple(b[:2]),tuple(b[2:]),color,1)
                cv2.putText(panel,str(r["id"]),tuple(b[:2]-[0,3]),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
            cv2.putText(panel,f"{name} | {(frame-1)/60:.2f}s | frame {frame}",(10,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
            panels.append(cv2.copyMakeBorder(panel,0,1,0,0,cv2.BORDER_CONSTANT))
        joined=np.hstack(panels);proc.stdin.write(joined.tobytes())
        if frame in [1,300,600,900,1200,1500,1800]:
            cv2.imwrite(str(out/f"review_{frame:04d}.jpg"),joined)
    proc.stdin.close();assert proc.wait()==0
    cap=cv2.VideoCapture(str(out/"comparison.mp4"));fps=cap.get(cv2.CAP_PROP_FPS);count=0
    while True:
        ok,_=cap.read()
        if not ok:break
        count+=1
    cap.release();assert count==1800 and abs(fps-60)<.01
    (out/"video_validation.json").write_text(json.dumps(dict(decoded_frames=count,fps=fps,duration=count/fps),indent=2))
    print("Validated",count,"frames",fps,"fps",flush=True)
    print(json.dumps(windows))


if __name__=="__main__":main()
