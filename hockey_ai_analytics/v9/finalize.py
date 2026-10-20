from pathlib import Path
import csv,json,cv2,numpy as np,subprocess,imageio_ffmpeg,configparser
ROOT=Path(__file__).resolve().parent;ff=imageio_ffmpeg.get_ffmpeg_exe();seqs=list(json.loads((ROOT/'selections.json').read_text()));summary=[]
for seq in seqs:
 out=ROOT/'results'/seq;metrics=json.loads((out/'metrics.json').read_text());metrics['hybrid_motion']=json.loads((out/'motion_metrics.json').read_text());metrics['revised']=json.loads((out/'revised_metrics.json').read_text());summary.append(metrics)
 data=ROOT/'data'/seq;cfg=configparser.ConfigParser();cfg.read(data/'seqinfo.ini');fps=float(cfg['Sequence']['frameRate']);gt=np.loadtxt(data/'gt/gt.txt',delimiter=',');target=metrics['selected_published_id'];tracks=list(csv.DictReader((out/'team_tracks.csv').open()));sam=list(csv.DictReader((out/'sam.csv').open()));hybrid=list(csv.DictReader((out/'revised.csv').open()));motion=list(csv.DictReader((out/'motion_checks.csv').open()));raw=out/'review_raw.mp4';writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1696,478))
 for f,(s,h) in enumerate(zip(sam,hybrid),1):
  native=cv2.imread(str(data/'img1'/f'{f:06d}.jpg'));H,W=native.shape[:2];im=cv2.resize(native,(848,477));panels=[]
  for label,row in [('ALL PLAYERS + SAM',s),('REVISED CONSENSUS',h)]:
   p=im.copy()
   if label.startswith('ALL'):
    for d in tracks:
     if int(d['frame'])!=f:continue
     x1,y1,x2,y2=(np.array([float(d[k]) for k in ['x1','y1','x2','y2']])*np.array([848/W,477/H]*2)).astype(int);color=(100,180,100) if d['team_hypothesis']=='LIGHT' else (220,150,100);cv2.rectangle(p,(x1,y1),(x2,y2),color,1)
   for g in gt[(gt[:,0]==f)&(gt[:,1]==target)]:
    x,y,w,hg=g[2:6];x1,y1,x2,y2=(np.array([x,y,x+w,y+hg])*np.array([848/W,477/H]*2)).astype(int);cv2.rectangle(p,(x1,y1),(x2,y2),(255,0,255),2)
   b=json.loads(row['box'])
   if b:
    x1,y1,x2,y2=(np.array(b)*np.array([848/W,477/H]*2)).astype(int);cv2.rectangle(p,(x1,y1),(x2,y2),(0,255,255),2)
   cv2.rectangle(p,(0,0),(848,62),(20,20,20),-1);cv2.putText(p,f'{seq} | {label} | f{f}',(8,17),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1);cv2.putText(p,f'QA: {row["state"]} | motion {motion[f-1]["status"]}',(8,36),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1);cv2.putText(p,'Magenta = published target reference; yellow = prediction',(8,54),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1);panels.append(cv2.copyMakeBorder(p,0,1,0,0,cv2.BORDER_CONSTANT))
  frame=np.hstack(panels);writer.write(frame)
  if f in [1,150,300]:cv2.imwrite(str(out/f'comparison_{f}.jpg'),frame)
 writer.release();dest=out/'review.mp4';subprocess.run([ff,'-y','-i',str(raw),'-c:v','libx264','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True,capture_output=True);cap=cv2.VideoCapture(str(dest));n=0
 while cap.read()[0]:n+=1
 cap.release();assert n==300;(out/'video_validation.json').write_text(json.dumps(dict(decoded_frames=n,width=1696,height=478,fps=fps)));print('Verified',seq,flush=True)
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
