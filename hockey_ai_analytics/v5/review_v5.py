"""Compare frozen predictions to the previous provisional reference; never feed reference to tracker."""
import sys,csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from make_review import load_video
from run_benchmark import write_csv
from detect_probe import in_ice
from PIL import Image,ImageDraw
import cv2,numpy as np


def center(r):return np.array([float(r['x'])+float(r['w'])/2,float(r['y'])+float(r['h'])/2])


def main():
    fs,pts=load_video(ROOT.parent/'evidence/source.mov')
    refs=list(csv.DictReader((ROOT.parent/'results/reference_ground_truth_PROVISIONAL.csv').open()))
    ps=list(csv.DictReader((ROOT/'results/target_v5_raw.csv').open()))
    all_objects=list(csv.DictReader((ROOT/'results/all_tracks.csv').open()))
    # Refined ice boundary excludes the foreground spectator; association itself
    # still runs on all detections, before this display/context filter.
    kept=[r for r in all_objects if in_ice([float(r['x']),float(r['y']),float(r['x'])+float(r['w']),float(r['y'])+float(r['h'])])]
    write_csv(ROOT/'results/on_ice_tracks.csv',kept)
    folder=ROOT/'review';folder.mkdir(exist_ok=True)
    per=[]
    for i,g in enumerate(refs):
        p=ps[i];eligible=i>=90 and g['status']=='visible';error=float(np.linalg.norm(center(p)-center(g))) if p['state']=='TRACKING' and eligible else ''
        per.append(dict(frame_index=i,pts_s=pts[i],reference_status=g['status'],provenance=g['provenance'],
            prediction_state=p['state'],center_error_px=error,identity_audit='PENDING',reason=p['reason']))
    write_csv(ROOT/'results/per_frame_comparison_PENDING_AUDIT.csv',per)
    eligible=[i for i in range(90,424) if refs[i]['status']=='visible']
    matched=[i for i in eligible if ps[i]['state']=='TRACKING']
    errors=[per[i]['center_error_px'] for i in matched]
    summary=dict(eligible_frames=len(eligible),matched=len(matched),coverage=len(matched)/len(eligible),
                 mean_center_error_px=float(np.mean(errors)),max_center_error_px=float(max(errors)),
                 above_20px=[i for i in matched if per[i]['center_error_px']>20],
                 pending_identity_audit=True,provisional_reference=True)
    (ROOT/'results/preliminary_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    for start in range(90,424,48):
        can=Image.new('RGB',(1200,8*190),'#eeeeee');d=ImageDraw.Draw(can)
        for j,i in enumerate(range(start,min(start+48,424))):
            cx,cy=center(refs[i]);left=int(cx)-50;top=int(cy)-40
            f=fs[i][top:top+85,left:left+100].copy()
            cv2.drawMarker(f,(round(cx-left),round(cy-top)),(0,200,0),cv2.MARKER_CROSS,7,1)
            p=ps[i]
            if p['state']=='TRACKING':
                pc=center(p);cv2.circle(f,tuple(np.round(pc-[left,top]).astype(int)),4,(255,0,255),1)
            x=j%6*200;y=j//6*190
            can.paste(Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)).resize((200,170)),(x,y))
            err=per[i]['center_error_px'];d.text((x+2,y+172),f'{i} '+(f'{err:.0f}px' if err!='' else p['state']),fill='black')
        can.save(folder/f'target_{start:03}.jpg',quality=95)
    # Full frames around the high-risk late crossing, with model IDs.
    for start,ids in [('crossing',[210,213,216,219,222,225]),('late',[345,354,360,366,375,390])]:
        can=Image.new('RGB',(1696,3*484),'white');d=ImageDraw.Draw(can)
        for j,i in enumerate(ids):
            f=fs[i].copy()
            for r in kept:
                if int(r['frame_index'])!=i:continue
                x,y,w,h=[float(r[k]) for k in ['x','y','w','h']];tid=int(r['track_id']);col=(255,0,255) if tid==4 else (0,0,255) if r['team_hypothesis']=='RED' else (200,100,0)
                cv2.rectangle(f,(round(x),round(y)),(round(x+w),round(y+h)),col,1);cv2.putText(f,str(tid),(round(x),round(y)-2),cv2.FONT_HERSHEY_SIMPLEX,.4,col,1)
            x=j%2*848;y=j//2*484;can.paste(Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)),(x,y));d.text((x+3,y+465),f'frame {i} : {ps[i]["state"]}',fill='black')
        can.save(folder/f'{start}.jpg',quality=95)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
