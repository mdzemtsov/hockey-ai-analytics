from pathlib import Path
import cv2,csv,json,subprocess,imageio_ffmpeg,hashlib
ROOT=Path(__file__).resolve().parent
rows=list(csv.DictReader((ROOT/'results/teams/team_tracks.csv').open()));selected=list(csv.DictReader((ROOT/'results/hockey/selected.csv').open()));out=ROOT/'results/teams';writer=cv2.VideoWriter(str(out/'team_review_raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(848,478));summaries=[]
for f in range(1,301):
 im=cv2.resize(cv2.imread(str(ROOT/'data/frames'/f'{f:06d}.jpg')),(848,477));objects=[r for r in rows if int(r['frame'])==f];counts={k:0 for k in ['LIGHT','DARK','REFEREE','UNKNOWN']}
 for r in objects:
  team=r['team_hypothesis'];counts[team]+=1;color={'LIGHT':(30,210,255),'DARK':(255,180,40),'REFEREE':(230,100,230),'UNKNOWN':(150,150,150)}[team]
  x1,y1,x2,y2=[int(float(r[k])*s) for k,s in zip(['x1','y1','x2','y2'],[848/1280,477/720]*2)]
  cv2.rectangle(im,(x1,y1),(x2,y2),color,1);cv2.putText(im,f'{team} {r["id"]}',(x1,max(12,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
 cv2.rectangle(im,(0,0),(848,48),(20,20,20),-1);cv2.putText(im,f'TEAM HYPOTHESES | frame {f} | light {counts["LIGHT"]} dark {counts["DARK"]}',(10,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
 cv2.putText(im,'Puck / zones / possession: NOT VALIDATED | tactical verdict: WITHHELD',(10,38),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
 writer.write(cv2.copyMakeBorder(im,0,1,0,0,cv2.BORDER_CONSTANT));summaries.append(dict(frame=f,**counts,puck_status='UNVALIDATED',rink_calibration='UNAVAILABLE',possession='UNKNOWN',tactical_verdict='WITHHELD'))
 if f==240:cv2.imwrite(str(out/'team_preview.jpg'),im)
writer.release()
with (out/'frame_context.csv').open('w') as h:w=csv.DictWriter(h,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
ff=imageio_ffmpeg.get_ffmpeg_exe()
for folder,name in [(ROOT/'results/baseline','review'),(ROOT/'results/hockey','review'),(out,'team_review_raw')]:
 src=folder/f'{name}.mp4';dest=folder/('team_review.mp4' if name=='team_review_raw' else 'review_h264.mp4')
 subprocess.run([ff,'-y','-i',str(src),'-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True,capture_output=True)
 cap=cv2.VideoCapture(str(dest));assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==300;assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))==848;cap.release()
print('Three H264 review videos verified: 300 frames each.')
