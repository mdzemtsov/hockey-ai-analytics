from pathlib import Path
import sys,csv,json,collections,subprocess
import cv2,numpy as np,imageio_ffmpeg
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT.parent/"v9"),str(ROOT.parent/"v7")]
from run_match import evaluate_selected,write_csv


def read(path):
    by=collections.defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            for k in ("frame","id"):r[k]=int(r[k])
            for k in ("x1","y1","x2","y2"):r[k]=float(r[k])
            by[r["frame"]].append(r)
    return by


def main():
    seq="CHI_VS_TOR_004";out=ROOT/"stable"/seq
    baseline=read(ROOT.parent/"v9/results"/seq/"tracks.csv")
    revised=read(out/"tracks.csv")
    gt=np.loadtxt(ROOT.parent/"v9/data"/seq/"gt/gt.txt",delimiter=",")
    selected=[]
    for f in range(1,301):
        r=next((r for r in revised[f] if r["id"]==1),None)
        selected.append([r[k] for k in ("x1","y1","x2","y2")] if r else None)
    rows,metrics=evaluate_selected(selected,gt,3)
    write_csv(out/"selected.csv",rows)
    (out/"selected_metrics.json").write_text(json.dumps(metrics,indent=2))
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    proc=subprocess.Popen([ff,"-y","-f","rawvideo","-vcodec","rawvideo","-pix_fmt","bgr24",
                           "-s","1696x478","-r","30","-i","-","-an","-c:v","libx264",
                           "-crf","20","-pix_fmt","yuv420p",str(out/"comparison.mp4")],
                          stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    for f in range(1,301):
        im=cv2.imread(str(ROOT.parent/"v9/data"/seq/"img1"/f"{f:06d}.jpg"))
        panels=[]
        for name,by in [("V9 baseline",baseline),("V10-C experimental",revised)]:
            panel=cv2.resize(im,(848,477))
            for r in by[f]:
                b=(np.array([r[k] for k in ("x1","y1","x2","y2")])*np.array([848/1280,477/720]*2)).astype(int)
                col=(0,255,255) if r["id"]==1 else (90,200,90)
                cv2.rectangle(panel,tuple(b[:2]),tuple(b[2:]),col,2)
                cv2.putText(panel,str(r["id"]),tuple(b[:2]-[0,3]),cv2.FONT_HERSHEY_SIMPLEX,.45,col,1)
            cv2.putText(panel,f"{name} | frame {f} | NOT VALIDATED",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
            panels.append(cv2.copyMakeBorder(panel,0,1,0,0,cv2.BORDER_CONSTANT))
        joined=np.hstack(panels);proc.stdin.write(joined.tobytes())
        if f in [58,63,66,68,90,141,250]:cv2.imwrite(str(out/f"review_{f:03d}.jpg"),joined)
    proc.stdin.close();assert proc.wait()==0
    cap=cv2.VideoCapture(str(out/"comparison.mp4"));count=0
    while True:
        ok,_=cap.read()
        if not ok:break
        count+=1
    cap.release();assert count==300
    (out/"video_validation.json").write_text(json.dumps(dict(decoded_frames=count,native_fps=30),indent=2))
    print(metrics)


if __name__=="__main__":main()
