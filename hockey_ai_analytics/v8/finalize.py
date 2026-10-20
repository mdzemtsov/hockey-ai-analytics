from pathlib import Path
import csv,json,cv2,numpy as np,subprocess,imageio_ffmpeg,hashlib
ROOT=Path(__file__).resolve().parent;V7=ROOT.parent/'v7';ff=imageio_ffmpeg.get_ffmpeg_exe()
for variant in ['bytetrack','high_detail']:
 out=ROOT/'results'/variant
 subprocess.run([ff,'-y','-i',str(out/'review.mp4'),'-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'review_h264.mp4')],check=True,capture_output=True)
gt=np.loadtxt(V7/'data/personnel_gt.txt',delimiter=',');sam=list(csv.DictReader((ROOT/'results/sam2/selected.csv').open()));hybrid=list(csv.DictReader((ROOT/'results/hybrid/selected.csv').open()));raw=ROOT/'results/selected_comparison_raw.mp4';writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),30,(1696,478))
for f,(a,b) in enumerate(zip(sam,hybrid),1):
 im=cv2.resize(cv2.imread(str(V7/'data/frames'/f'{f:06d}.jpg')),(848,477));panels=[]
 for label,row in [('SAM 2.1',a),('SAM + detector + team + stability',b)]:
  panel=im.copy();truth=gt[(gt[:,0]==f)&(gt[:,1]==1),2:6].copy();truth[:,2:]+=truth[:,:2]
  for t in truth:
   x1,y1,x2,y2=(t*np.array([848/1280,477/720]*2)).astype(int);cv2.rectangle(panel,(x1,y1),(x2,y2),(255,0,255),2)
  box=json.loads(row['box'])
  if box:
   x1,y1,x2,y2=(np.array(box)*np.array([848/1280,477/720]*2)).astype(int);cv2.rectangle(panel,(x1,y1),(x2,y2),(0,255,255),2)
  cv2.rectangle(panel,(0,0),(848,48),(20,20,20),-1);cv2.putText(panel,f'{label} | frame {f}: {row["state"]}',(8,18),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1);cv2.putText(panel,'QA: magenta = published reference; yellow = prediction',(8,39),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
  panels.append(cv2.copyMakeBorder(panel,0,1,0,0,cv2.BORDER_CONSTANT))
 frame=np.hstack(panels);writer.write(frame)
 if f==240:cv2.imwrite(str(ROOT/'results/comparison_240.jpg'),frame)
writer.release();dest=ROOT/'results/selected_comparison.mp4';subprocess.run([ff,'-y','-i',str(raw),'-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True,capture_output=True)
for p in [dest,*ROOT.glob('results/*/review_h264.mp4')]:
 cap=cv2.VideoCapture(str(p));count=0
 while cap.read()[0]:count+=1
 cap.release();assert count==300,(p,count)
print('Three H264 videos fully decoded; 300 frames each.')
