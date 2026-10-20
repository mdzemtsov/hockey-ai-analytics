"""Publish provisional reference, audit tables, metrics and reproducible review media.

Reference positions between manual anchors remain reviewed interpolation, NOT
independent manually clicked ground truth. Never silently promote their provenance.
"""
import csv
import json
import math
import subprocess
from statistics import mean
from pathlib import Path
import cv2
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw
from make_review import ROOT,load_video
from run_benchmark import write_csv

UNCERTAIN=set(range(213,220))|set(range(357,367))


def center(r):return np.array([float(r['x'])+float(r['w'])/2,float(r['y'])+float(r['h'])/2])


def metrics(reference,pred,anchors_only=False):
    eligible=[i for i,g in enumerate(reference) if i>=90 and g['status']=='visible' and (not anchors_only or g['provenance']=='manual')]
    covered=[i for i in eligible if pred[i]['state']=='TRACKING']
    errors=[float(np.linalg.norm(center(reference[i])-center(pred[i]))) for i in covered]
    all_track=[i for i,p in enumerate(pred) if p['state']=='TRACKING']
    unknown=[i for i in all_track if i in UNCERTAIN]
    wrong=sum(bool(pred[i].get('observed_id')) and pred[i]['observed_id']!='RED_TEST_01' for i in all_track)
    switches=0;last=None
    for p in pred:
        identity=p.get('observed_id') or None
        if identity and last and identity!=last:switches+=1
        last=identity
    return dict(reference_scope='manual_anchors' if anchors_only else 'provisional_reviewed_interpolation',
        eligible_post_seed_frames=len(eligible),predictions_on_eligible_frames=len(covered),
        coverage=len(covered)/len(eligible),missing_visible_frames=len(eligible)-len(covered),
        mean_center_error_px_conditional=mean(errors) if errors else None,
        max_center_error_px_conditional=max(errors) if errors else None,
        fraction_within_half_player_height_all_eligible=sum(e<=float(reference[i]['h'])/2 for e,i in zip(errors,covered))/len(eligible),
        tracking_state_frames=len(all_track),identity_reviewed_visible_predictions=len(all_track)-len(unknown),
        identity_uncertain_predictions=len(unknown),confirmed_other_player_frames=wrong,
        observed_identity_switches_on_reviewed_intervals=switches,
        identity_certification='ASSISTANT_VISUAL_AUDIT_ONLY; HUMAN_REVIEW_PENDING',
        release='FAIL_INCOMPLETE_TRACK; NOT_VALIDATED')


def media(frames,pts,refs,p3,p4):
    folder=ROOT/'results/render_frames';folder.mkdir(exist_ok=True)
    review=ROOT/'evidence/review'
    for start in range(90,424,48):
        can=Image.new('RGB',(1200,8*160),'#eeeeee');draw=ImageDraw.Draw(can)
        for j,i in enumerate(range(start,min(start+48,424))):
            cx,cy=center(refs[i]);left=int(cx)-50;top=int(cy)-35
            crop=frames[i][top:top+70,left:left+100].copy()
            cv2.drawMarker(crop,(round(cx-left),round(cy-top)),(0,200,0),cv2.MARKER_CROSS,7,1)
            info=[]
            for name,p,color in [('3',p3[i],(0,220,255)),('4',p4[i],(255,0,255))]:
                if p['state']=='TRACKING':
                    pc=center(p);cv2.circle(crop,tuple(np.round(pc-[left,top]).astype(int)),4,color,1)
                    info.append(f'{name}:{np.linalg.norm(pc-[cx,cy]):.0f}px')
                else:info.append(f'{name}:L')
            x=j%6*200;y=j//6*160
            can.paste(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)).resize((200,140)),(x,y))
            draw.text((x+2,y+141),f'{i} '+','.join(info),fill='black')
        can.save(review/f'final_compare_{start:03}.jpg',quality=95)
    # Three synchronized full-resolution panels; reference is explicitly provisional.
    for i,frame in enumerate(frames):
        panels=[]
        for label,r,color in [('REF: AI-REVIEWED / HUMAN PENDING',refs[i],(0,200,0)),('V3: DIGITIZED MARKER',p3[i],(0,220,255)),('V4: EXPERIMENTAL',p4[i],(255,0,255))]:
            f=frame.copy();cv2.rectangle(f,(0,0),(848,68),(25,25,25),-1)
            cv2.putText(f,label,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.52,color,1,cv2.LINE_AA)
            state=r.get('state',r.get('status',''))
            cv2.putText(f,f'frame={i} PTS={pts[i]:.5f}s {state}',(8,41),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
            if label.startswith('REF'):
                cv2.putText(f,r['provenance'],(8,60),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
            if state in ('visible','TRACKING'):
                pc=center(r);cv2.drawMarker(f,tuple(np.round(pc).astype(int)),color,cv2.MARKER_CROSS,12,2)
                if label.startswith('REF'):
                    x,y,w,h=[float(r[k]) for k in ['x','y','w','h']]
                    cv2.rectangle(f,(round(x),round(y)),(round(x+w),round(y+h)),color,1)
            panels.append(f)
        cv2.imwrite(str(folder/f'{i:04}.png'),np.concatenate(panels,axis=1))
    timeline=[]
    for i in range(len(frames)):
        duration=pts[i+1]-pts[i] if i+1<len(pts) else 1/30
        timeline.extend([f"file '{i:04}.png'",'option framerate 60000',f'duration {duration:.10f}'])
    (folder/'timeline.txt').write_text('\n'.join(timeline)+'\n')
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    cmd=[ffmpeg,'-y','-v','warning','-f','concat','-safe','0','-i',str(folder/'timeline.txt'),
         '-i',str(ROOT/'evidence/source.mov'),'-map','0:v:0','-map','1:a:0?',
         '-fps_mode','vfr','-enc_time_base','1:60000','-video_track_timescale','60000',
         '-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',str(ROOT/'results/qa_comparison.mp4')]
    res=subprocess.run(cmd,capture_output=True,text=True)
    (ROOT/'results/render_log.txt').write_text(res.stderr)
    if res.returncode:raise RuntimeError(res.stderr)


def main():
    fs,pts=load_video(ROOT/'evidence/source.mov')
    refs=list(csv.DictReader((ROOT/'reference_proposals.csv').open()))
    for i,r in enumerate(refs):
        if r['provenance']=='interpolated':r['provenance']='reviewed_interpolation'
        r['reviewer']='assistant_visual_review_NOT_HUMAN'
        r['notes']='All-frame contact-sheet review; coarse bbox; independent human review pending'
        if i in UNCERTAIN:r.update(status='occluded',notes='Partial overlap; localization/identity excluded from scores')
    write_csv(ROOT/'results/reference_ground_truth_PROVISIONAL.csv',refs)
    preds={name:list(csv.DictReader((ROOT/'results'/file).open())) for name,file in [('V3','v3_digitized.csv'),('V4','v4_predictions_raw.csv')]}
    decisions=list(csv.DictReader((ROOT/'identity_review_intervals.csv').open()))
    detail=[];summary={}
    for name,pred in preds.items():
        for i,p in enumerate(pred):
            audit='NO_PREDICTION'
            if p['state']=='TRACKING':
                matches=[r for r in decisions if r['tracker']==name and int(r['start_frame'])<=i<=int(r['end_frame'])]
                if len(matches)!=1:raise ValueError('Missing or overlapping identity audit interval')
                audit=matches[0]['observed_identity']
            p['observed_id']='RED_TEST_01' if audit=='RED_TEST_01' else ''
            detail.append(dict(tracker=name,frame_index=i,pts_s=pts[i],reference_status=refs[i]['status'],
                provenance=refs[i]['provenance'],prediction_state=p['state'],identity_audit=audit,
                center_error_px=float(np.linalg.norm(center(refs[i])-center(p))) if p['state']=='TRACKING' and refs[i]['status']=='visible' else '',
                reviewer='assistant_visual_review_NOT_HUMAN'))
        summary[name]={'dense_provisional':metrics(refs,pred),'manual_anchor_only':metrics(refs,pred,True)}
        write_csv(ROOT/f'results/{name.lower()}_audited.csv',pred)
    write_csv(ROOT/'results/per_frame_qa.csv',detail)
    shared=[i for i,g in enumerate(refs) if i>=90 and g['status']=='visible' and all(p[i]['state']=='TRACKING' for p in preds.values())]
    summary['paired_same_frames']={'n':len(shared),**{n:mean(float(np.linalg.norm(center(refs[i])-center(p[i]))) for i in shared) for n,p in preds.items()}}
    (ROOT/'results/comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_csv(ROOT/'results/comparison.csv',[dict(tracker=n,**s['dense_provisional']) for n,s in summary.items() if n in preds])
    import sys
    if '--skip-media' not in sys.argv:media(fs,pts,refs,preds['V3'],preds['V4'])
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
