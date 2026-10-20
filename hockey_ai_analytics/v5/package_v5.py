"""Create V5 review artifacts and geometric observations, with interpretation blocked."""
import os,sys,csv,json,hashlib,subprocess
from pathlib import Path
from collections import Counter,defaultdict
from statistics import mean
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import cv2,numpy as np,imageio_ffmpeg
from make_review import load_video
from run_benchmark import write_csv
from target_gate import TargetGate


def center(r):return np.array([float(r['x'])+float(r['w'])/2,float(r['y'])+float(r['h'])/2])


def runs(indices):
    out=[]
    for i in indices:
        if not out or i!=out[-1][-1]+1:out.append([i])
        else:out[-1].append(i)
    return out


def main():
    out=ROOT/'results'
    frames,pts=load_video(ROOT.parent/'evidence/source.mov')
    pred=list(csv.DictReader((out/'target_v5_raw.csv').open()))
    record=json.loads((out/'visual_audit_record.json').read_text())
    assert record['prediction_sha256']==hashlib.sha256((out/'target_v5_raw.csv').read_bytes()).hexdigest(),'Predictions changed: repeat visual audit'
    assert record['source_sha256']==hashlib.sha256((ROOT.parent/'evidence/source.mov').read_bytes()).hexdigest(),'Source changed: repeat visual audit'
    refs=list(csv.DictReader((ROOT.parent/'results/reference_ground_truth_PROVISIONAL.csv').open()))
    tracks=list(csv.DictReader((out/'on_ice_tracks.csv').open()))
    grouped=defaultdict(list)
    for r in tracks:grouped[int(r['frame_index'])].append(r)
    assert len(pred)==len(frames)==424
    audit=[];eligible=[];matched=[];errors=[]
    for i,(g,p) in enumerate(zip(refs,pred)):
        ok=i>=90 and g['status']=='visible'
        if ok:eligible.append(i)
        error='';identity='NO_PREDICTION'
        if p['state']=='TRACKING':
            identity='RED_TEST_01' if g['status']=='visible' else 'IDENTITY_UNCERTAIN'
            if ok:
                matched.append(i);error=float(np.linalg.norm(center(g)-center(p)));errors.append(error)
        audit.append(dict(frame_index=i,pts_s=pts[i],state=p['state'],reference_status=g['status'],
            reference_provenance=g['provenance'],center_error_px=error,observed_identity=identity,
            reviewer='ASSISTANT_VISUAL_REVIEW_NOT_INDEPENDENT_HUMAN',reason=p['reason']))
    write_csv(out/'target_identity_audit.csv',audit)
    byframe_counts=[len(grouped[i]) for i in range(424)]
    missing=[i for i in range(90,424) if pred[i]['state']!='TRACKING']
    breaks=[dict(start_frame=r[0],end_frame=r[-1],start_pts_s=pts[r[0]],end_pts_s=pts[r[-1]],frames=len(r)) for r in runs(missing)]
    base=json.loads((ROOT.parent/'results/comparison.json').read_text())
    comparison=[]
    for name in ['V3','V4']:
        s=base[name]['dense_provisional'];comparison.append(dict(version=name,eligible_frames=s['eligible_post_seed_frames'],matched=s['predictions_on_eligible_frames'],coverage=s['coverage'],mean_center_error_px=s['mean_center_error_px_conditional']))
    comparison.append(dict(version='V5',eligible_frames=len(eligible),matched=len(matched),coverage=len(matched)/len(eligible),mean_center_error_px=mean(errors)))
    write_csv(out/'version_comparison.csv',comparison)
    manual=[i for i in eligible if refs[i]['provenance']=='manual'];mhit=[i for i in manual if pred[i]['state']=='TRACKING']
    # Re-run only the output gate on cached detections. This tests reproducible
    # decisions; it does NOT claim a second detector inference is identical.
    replay=TargetGate()
    for i in range(424):
        objects=[]
        for r in grouped[i]:
            q=dict(r)
            for k in ['x','y','w','h','detector_conf']:q[k]=float(q[k])
            q['track_id']=int(q['track_id']);objects.append(q)
        rr=replay.update(i,pts[i],objects)
        assert rr['state']==pred[i]['state'],(i,'gate replay state differs')
        for k in ['x','y','w','h']:
            assert str(rr[k])==pred[i][k],(i,k,'gate replay box differs')
    summary=dict(version='V5',states=dict(Counter(p['state'] for p in pred)),
        eligible_frames=len(eligible),matched_visible_frames=len(matched),coverage=len(matched)/len(eligible),
        mean_center_error_px_conditional=mean(errors),max_center_error_px=max(errors),
        manual_anchor_eligible=len(manual),manual_anchor_matched=len(mhit),
        manual_anchor_mean_center_error_px=mean(float(audit[i]['center_error_px']) for i in mhit),
        confirmed_wrong_target_frames_on_reviewed_visible_intervals=record['confirmed_target_switches_on_unambiguous_reviewed_intervals'],
        observed_target_identity_transitions_on_reviewed_visible_intervals=record['confirmed_target_switches_on_unambiguous_reviewed_intervals'],
        predictions_on_uncertain_reference_frames=sum(a['observed_identity']=='IDENTITY_UNCERTAIN' for a in audit),
        all_other_ids='UNVERIFIED; no multi-object IDF1/HOTA claimed',
        tracked_model_id_hypotheses=len(set(r['track_id'] for r in tracks)),
        median_person_hypotheses_per_frame=float(np.median(byframe_counts)),
        min_person_hypotheses_per_frame=min(byframe_counts),max_person_hypotheses_per_frame=max(byframe_counts),
        breaks=breaks,ends_tracking=pred[-1]['state']=='TRACKING',
        ground_truth='30 manual assistant anchors + reviewed interpolation; provisional',
        gate_replay_identical=True,puck_tracking=False,possession=False,tactical_inference=False,
        release='DEVELOPMENT_PROGRESS_NOT_RELEASE_VALIDATED')
    (out/'qa_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Conservative scene context. Screen distances cannot be interpreted as rink metres.
    context=[]
    for i,p in enumerate(pred):
        near=[]
        if p['state']=='TRACKING':
            near=sorted([dict(model_track_id=int(r['track_id']),screen_distance_px=float(np.linalg.norm(center(p)-center(r))),team_hypothesis=r['team_hypothesis']) for r in grouped[i] if r['track_id']!=p['track_id']],key=lambda x:x['screen_distance_px'])[:3]
        context.append(dict(frame_index=i,pts_s=pts[i],target_state=p['state'],nearby_person_hypotheses=near,
            puck=None,possession=None,play_phase=None,coaching_verdict=None,
            blocked_reason='NO_VALIDATED_PUCK_POSSESSION_OR_ROLE_IDENTITIES'))
    (out/'scene_context.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in context))
    # Observed reversal in screen-space only, selected algorithmically, not a tactical judgement.
    observations=[]
    for i in range(110,404):
        if pred[i]['state']!='TRACKING':continue
        before=[center(pred[j])[0] for j in range(i-20,i) if pred[j]['state']=='TRACKING']
        after=[center(pred[j])[0] for j in range(i+1,i+21) if pred[j]['state']=='TRACKING']
        if len(before)>=12 and len(after)>=12 and center(pred[i])[0]>max(np.median(before),np.median(after))+4:
            observations.append(dict(frame_index=i,pts_s=pts[i],observation='SCREEN_RIGHT_TO_LEFT_TURN_CANDIDATE',interpretation='NO_TACTICAL_OR_INTENT_CLAIM',review_status='ASSISTANT_VISUAL_SEQUENCE_REVIEW'))
    if observations:
        # One representative peak, not many copies of the same observation.
        representative=max(observations,key=lambda r:center(pred[r['frame_index']])[0])
        write_csv(out/'geometric_observations.csv',[representative])
    else:(out/'geometric_observations.csv').write_text('frame_index,pts_s,observation,interpretation,review_status\n')
    if '--no-render' not in sys.argv:render(frames,pts,pred,grouped)
    print(json.dumps(summary,indent=2))


def render(frames,pts,pred,grouped):
    out=ROOT/'results';tmp=out/'render_frames';tmp.mkdir(exist_ok=True)
    for i,frame in enumerate(frames):
        left=frame.copy();right=frame.copy();p=pred[i]
        for r in grouped[i]:
            x,y,w,h=[float(r[k]) for k in ['x','y','w','h']]
            col=(80,80,230) if r['team_hypothesis']=='RED' else (220,170,80)
            cv2.rectangle(left,(round(x),round(y)),(round(x+w),round(y+h)),col,1)
            cv2.putText(left,r['track_id'],(round(x),max(72,round(y)-2)),cv2.FONT_HERSHEY_SIMPLEX,.36,col,1,cv2.LINE_AA)
        if p['state']=='TRACKING':
            x,y,w,h=[float(p[k]) for k in ['x','y','w','h']]
            cv2.rectangle(right,(round(x),round(y)),(round(x+w),round(y+h)),(255,0,255),2)
            cv2.putText(right,'RED_TEST_01',(round(x),round(y)-4),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,0,255),1,cv2.LINE_AA)
        for panel,label in [(left,'MULTI-OBJECT: model IDs are unverified'),(right,'V5: automatic target / not human-certified')]:
            cv2.rectangle(panel,(0,0),(848,67),(20,20,20),-1)
            cv2.putText(panel,label,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.49,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(panel,f'frame {i}  PTS {pts[i]:.5f}s  target {p["state"]}',(8,39),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(panel,'Puck / possession / tactical judgement: UNKNOWN',(8,58),cv2.FONT_HERSHEY_SIMPLEX,.42,(150,200,255),1,cv2.LINE_AA)
        both=np.concatenate([left,right],axis=1);cv2.imwrite(str(tmp/f'{i:04}.png'),both)
        if i in [90,216,221,360,390,423]:cv2.imwrite(str(ROOT/f'review/render_{i}.jpg'),both)
    lines=[]
    for i in range(424):lines.extend([f"file '{i:04}.png'",'option framerate 60000',f'duration {(pts[i+1]-pts[i] if i<423 else 1/30):.10f}'])
    (tmp/'timeline.txt').write_text('\n'.join(lines)+'\n')
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff,'-y','-v','error','-f','concat','-safe','0','-i',str(tmp/'timeline.txt'),'-i',str(ROOT.parent/'evidence/source.mov'),
        '-map','0:v:0','-map','1:a:0?','-fps_mode','vfr','-enc_time_base','1:60000','-video_track_timescale','60000',
        '-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',str(out/'v5_multi_target.mp4')],check=True)
    decoded,outpts=load_video(out/'v5_multi_target.mp4')
    assert len(decoded)==424
    maxerr=max(abs(a-b) for a,b in zip(pts,outpts));assert maxerr<.001
    audio=[]
    for file in [ROOT.parent/'evidence/source.mov',out/'v5_multi_target.mp4']:
        rr=subprocess.run([ff,'-v','error','-i',str(file),'-map','0:a:0','-c','copy','-f','hash','-hash','sha256','-'],capture_output=True,text=True,check=True);audio.append(rr.stdout.strip())
    assert audio[0]==audio[1]
    (out/'render_validation.json').write_text(json.dumps(dict(decoded_frames=424,max_pts_error_s=maxerr,audio_packet_hash_equal=True),indent=2)+'\n')


if __name__=='__main__':main()
