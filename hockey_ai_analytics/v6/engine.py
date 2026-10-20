"""Local video preparation, inference, review intervals and rendering. No tactical claims."""
import os,json,csv,time,hashlib,subprocess,sys,threading
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'runtime')
os.environ['MPLCONFIGDIR']=str(ROOT/'runtime/matplotlib')
os.environ['YOLO_AUTOINSTALL']='false'
import cv2,numpy as np,imageio_ffmpeg
from gate import ManualGate,POLICY
MODEL_LOCK=threading.Lock()
DEMO_HASH='9ba5b41500033edc107f808ce9d56188b2beaabbf4e8d7cdf65b9fa951e27ce8'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def prepare(path,folder,progress):
    cap=cv2.VideoCapture(str(path));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));pts=[]
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not cap.isOpened() or width<16 or height<16:raise ValueError('Не удалось прочитать видео. Попробуйте MOV или MP4.')
    while True:
        ok,frame=cap.read()
        if not ok:break
        pts.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
        if len(pts)%150==0:progress('prepare',min(.5,len(pts)/max(count,1)*.5),'Проверяю кадры видео')
        if pts[-1]>1200:raise ValueError('Для этого пилота выберите эпизод длительностью до 20 минут.')
    cap.release()
    if not pts:raise ValueError('В видео нет доступных кадров.')
    # Some decoders do not expose timestamps; make the limitation explicit.
    timestamp_source='decoded_pts'
    if len(pts)>1 and any(b<=a for a,b in zip(pts,pts[1:])):raise ValueError('Временные метки видео неоднозначны. Нужен экспорт с корректными метками времени.')
    progress('prepare',.55,'Готовлю видео для просмотра')
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff,'-y','-v','error','-i',str(path),'-map','0:v:0','-an','-vf','scale=w=trunc(min(1280\\,iw)/2)*2:h=-2',
        '-fps_mode','passthrough','-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'preview.mp4')],check=True,capture_output=True)
    meta=dict(width=width,height=height,frame_count=len(pts),pts=pts,duration=pts[-1]+(pts[-1]-pts[-2] if len(pts)>1 else 1/30),
        timestamp_source=timestamp_source,sha256=digest(path))
    (folder/'meta.json').write_text(json.dumps(meta)+'\n')
    return meta


def colour(frame,box):
    x,y,w,h=box
    crop=frame[max(0,int(y+.2*h)):int(y+.68*h),max(0,int(x+.2*w)):int(x+.8*w)]
    if crop.size==0:return 'NONRED_UNKNOWN_ROLE',0.
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    red=(((hsv[:,:,0]<13)|(hsv[:,:,0]>165))&(hsv[:,:,1]>85)&(hsv[:,:,2]>40)).mean()
    return ('RED' if red>.13 else 'NONRED_UNKNOWN_ROLE'),float(red)


def detection_signature(meta):
    from importlib.metadata import version
    files={'engine':Path(__file__), 'model':ROOT.parent/'v5/models/yolo11s.pt',
           'tracker':ROOT.parent/'v5/botsort_reid.yaml'}
    if meta['sha256']==DEMO_HASH:
        files['legacy_detections']=ROOT.parent/'v5/results/on_ice_tracks.csv'
    return dict(schema=1,source_sha256=meta['sha256'],frame_count=meta['frame_count'],
                files={k:digest(p) if p.exists() else None for k,p in files.items()},
                packages={k:version(k) for k in ['torch','ultralytics','numpy','opencv-python']})


def read_cache(cache,signature):
    try:
        saved=json.loads(cache.read_text())
        if not isinstance(saved,dict) or saved.get('signature')!=signature:return None
        frames=saved['frames']
        if len(frames)!=signature['frame_count']:return None
        payload=json.dumps(frames,sort_keys=True).encode()
        if hashlib.sha256(payload).hexdigest()!=saved['frames_sha256']:return None
        return frames
    except (OSError,ValueError,KeyError,TypeError):return None


def save_cache(cache,signature,frames):
    saved=dict(signature=signature,frames=frames,frames_sha256=hashlib.sha256(json.dumps(frames,sort_keys=True).encode()).hexdigest())
    tmp=cache.with_suffix('.tmp');tmp.write_text(json.dumps(saved));tmp.replace(cache)


def detect(path,meta,cache,progress,cancel):
    signature=detection_signature(meta)
    cached=read_cache(cache,signature)
    if cached is not None:return cached,'validated_video_detection_cache'
    if meta['sha256']==DEMO_HASH:
        known=ROOT.parent/'v5/results/on_ice_tracks.csv'
        if known.exists():
            grouped=[[] for _ in range(meta['frame_count'])]
            for row in csv.DictReader(known.open()):
                o=dict(row)
                for k in ['x','y','w','h','detector_conf','red_fraction']:o[k]=float(o[k])
                o['track_id']=int(o['track_id']);grouped[int(row['frame_index'])].append(o)
            save_cache(cache,signature,grouped);return grouped,'verified_source_hash_v5_neural_cache'
    progress('detect',0,'Загружаю локальную модель')
    import torch
    from ultralytics import YOLO,settings
    settings.update({k:False for k in ['sync','wandb','mlflow','comet'] if k in settings})
    torch.set_num_threads(4);torch.manual_seed(0);cv2.setNumThreads(1);cv2.setRNGSeed(0)
    model=YOLO(str(ROOT.parent/'v5/models/yolo11s.pt'))
    grouped=[];cap=cv2.VideoCapture(str(path));i=0
    while True:
        if cancel.is_set():cap.release();raise InterruptedError('Обработка отменена.')
        ok,frame=cap.read()
        if not ok:break
        res=model.track(frame,persist=True,tracker=str(ROOT.parent/'v5/botsort_reid.yaml'),classes=[0],imgsz=1280,conf=.10,iou=.55,device='cpu',verbose=False)[0]
        objects=[]
        if res.boxes.id is not None:
            for box,conf,tid in zip(res.boxes.xyxy.cpu().numpy(),res.boxes.conf.cpu().numpy(),res.boxes.id.cpu().numpy()):
                x,y,z,t=map(float,box);w=z-x;h=t-y
                if min(w,h)<4:continue
                label,fraction=colour(frame,[x,y,w,h])
                objects.append(dict(track_id=int(tid),x=x,y=y,w=w,h=h,detector_conf=float(conf),team_hypothesis=label,red_fraction=fraction))
        grouped.append(objects);i+=1
        if i%15==0:progress('detect',i/meta['frame_count'],f'Обрабатываю кадр {i} из {meta["frame_count"]}')
    cap.release()
    if len(grouped)!=meta['frame_count']:raise ValueError('При повторном чтении изменилось число кадров. Результат не выдан.')
    save_cache(cache,signature,grouped);return grouped,'new_local_neural_inference'


def analyse(path,meta,video_folder,folder,selections,progress,cancel):
    with MODEL_LOCK:
        if cancel.is_set():raise InterruptedError('Обработка отменена.')
        grouped,mode=detect(path,meta,video_folder/'detections.json',progress,cancel)
        provenance=dict(policy=POLICY,detection=detection_signature(meta),
            code_sha256={p.name:digest(p) for p in [Path(__file__),ROOT/'gate.py']},
            experimental_identity_models_enabled=False,independent_identity_validation=False)
        (folder/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        gate=ManualGate(selections);rows=[]
        for i,objects in enumerate(grouped):rows.append(gate.update(i,meta['pts'][i],objects))
        (folder/'selection.json').write_text(json.dumps(dict(source_sha256=meta['sha256'],selections=selections,binding_events=gate.events),indent=2)+'\n')
        write_csv(folder/'target.csv',rows)
        failed=[e for e in gate.events if e['error']]
        if failed:
            return dict(state='needs_selection',message='В рамке не удалось однозначно найти фигуру. Уточните рамку или выберите соседний кадр.',binding_events=gate.events)
        progress('render',0,'Готовлю результат сопровождения')
        render(path,meta,folder,rows,cancel,progress)
        intervals=[]
        for r in rows:
            i=r['frame_index']
            if i<gate.first or r['state']=='TRACKING':continue
            if intervals and i==intervals[-1]['end_frame']+1:intervals[-1]['end_frame']=i
            else:intervals.append(dict(start_frame=i,end_frame=i,confirmed=False))
        for r in intervals:
            r.update(start_s=meta['pts'][r['start_frame']],end_s=meta['pts'][r['end_frame']])
            confirmations=[s['frame_index'] for s in selections[1:] if s['confirmed_same_player'] and s['frame_index']==r['end_frame']+1]
            r.update(confirmed=bool(confirmations),confirmation_frame=confirmations[0] if confirmations else None)
        tracked=sum(r['state']=='TRACKING' for r in rows);window=len(rows)-gate.first
        report=dict(state='ready',source_sha256=meta['sha256'],inference_mode=mode,frame_count=len(rows),
            first_frame=gate.first,frames_with_marker=tracked,frames_after_selection=window,marker_fraction=tracked/window,
            review_intervals=intervals,manual_confirmations=len(selections)-1,ends_with_marker=rows[-1]['state']=='TRACKING',
            policy=POLICY,unknown_frames=sum(r['state']=='UNKNOWN' for r in rows),
            identity_accuracy=None,identity_status='USER_SELECTED_START_AUTOMATIC_TRACK_REQUIRES_REVIEW',
            puck_tracking=False,possession=False,tactical_analysis=False,
            message='Обработка завершена. Неопределённые участки требуют повторного выбора игрока. Точность личности не измерена.')
        (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        return report


def render(path,meta,folder,rows,cancel,progress):
    ff=imageio_ffmpeg.get_ffmpeg_exe();cap=cv2.VideoCapture(str(path))
    # Disk-backed frames and a VFR timeline preserve original timestamp gaps.
    tmp=folder/'frames';tmp.mkdir(exist_ok=True)
    try:
        for i,r in enumerate(rows):
            if cancel.is_set():raise InterruptedError('Обработка отменена.')
            ok,f=cap.read()
            if not ok:raise ValueError('Не удалось отрисовать все кадры.')
            if r['state']=='TRACKING':
                x,y,w,h=[float(r[k]) for k in ['x','y','w','h']]
                cv2.rectangle(f,(round(x),round(y)),(round(x+w),round(y+h)),(0,205,255),2)
            cv2.rectangle(f,(0,0),(meta['width'],min(46,meta['height'])),(20,20,20),-1)
            label='PLAYER HYPOTHESIS' if r['state']=='TRACKING' else 'UNKNOWN - SELECT AGAIN' if i>=rows[0]['frame_index'] and r['state']!='NOT_STARTED' else 'BEFORE SELECTION'
            cv2.putText(f,f'{label} | {meta["pts"][i]:.3f}s',(8,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(f,'Automatic tracking; identity review required',(8,38),cv2.FONT_HERSHEY_SIMPLEX,.4,(180,210,240),1,cv2.LINE_AA)
            cv2.imwrite(str(tmp/f'{i:06}.jpg'),f,[cv2.IMWRITE_JPEG_QUALITY,94])
            if i%60==0:progress('render',i/len(rows)*.8,'Готовлю видео результата')
        cap.release();lines=[]
        for i in range(len(rows)):
            dur=meta['pts'][i+1]-meta['pts'][i] if i+1<len(rows) else meta['duration']-meta['pts'][i]
            lines.extend([f"file '{i:06}.jpg'",'option framerate 60000',f'duration {dur:.10f}'])
        (tmp/'timeline.txt').write_text('\n'.join(lines)+'\n')
        subprocess.run([ff,'-y','-v','error','-f','concat','-safe','0','-i',str(tmp/'timeline.txt'),'-i',str(path),
            '-map','0:v:0','-map','1:a:0?','-fps_mode','vfr','-enc_time_base','1:60000','-video_track_timescale','60000',
            '-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k','-movflags','+faststart',str(folder/'result.mp4')],check=True,capture_output=True)
        # Preserve the final frame's display duration as well as its presentation timestamp.
        tail_ticks=round((meta['duration']-meta['pts'][-1])*60000)
        last_ticks=round(meta['pts'][-1]*60000)
        duration_filter=f'setts=pts=PTS:dts=DTS:duration=if(gte(PTS\\,{last_ticks-2})\\,{tail_ticks}\\,DURATION)'
        subprocess.run([ff,'-y','-v','error','-i',str(folder/'result.mp4'),'-map','0','-c','copy',
            '-bsf:v',duration_filter,'-video_track_timescale','60000','-movflags','+faststart',str(folder/'timed.mp4')],check=True,capture_output=True)
        (folder/'timed.mp4').replace(folder/'result.mp4')
        check=cv2.VideoCapture(str(folder/'result.mp4'));n=0;maxerr=0
        while True:
            ok,_=check.read()
            if not ok:break
            if n>=len(rows):raise ValueError('Лишние кадры в результате.')
            maxerr=max(maxerr,abs(check.get(cv2.CAP_PROP_POS_MSEC)/1000-meta['pts'][n]));n+=1
        check.release()
        if n!=len(rows) or maxerr>.001:raise ValueError('Проверка кадров или временных меток результата не прошла.')
        (folder/'render_validation.json').write_text(json.dumps(dict(decoded_frames=n,max_pts_error_s=maxerr,audio='AAC copy of source content, reencoded for browser compatibility'))+'\n')
    finally:
        cap.release()
        import shutil
        shutil.rmtree(tmp,ignore_errors=True)
