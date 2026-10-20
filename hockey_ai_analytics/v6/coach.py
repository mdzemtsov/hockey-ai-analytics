"""Evidence-first coaching prototype. Image geometry is not rink geometry."""
import json,math,csv,subprocess,shutil
from pathlib import Path
import cv2,numpy as np
from engine import ROOT,MODEL_LOCK,digest
VERSION='coach-evidence-v1.2'
CODE_SHA=digest(Path(__file__))
MODEL=ROOT.parent/'v7/data/HockeyAI.pt'
SOURCE='https://www.usahockey.com/news_article/show/775908'
COLORS=['RED','BLUE','YELLOW','GREEN','WHITE','DARK']

def jersey_color(frame,box):
    x,y,w,h=box;crop=frame[max(0,int(y+.22*h)):max(0,int(y+.58*h)),max(0,int(x+.25*w)):max(0,int(x+.75*w))]
    if crop.size<60:return 'UNKNOWN',0.
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV);H,S,V=cv2.split(hsv)
    masks=[((H<12)|(H>168))&(S>95)&(V>65),(H>95)&(H<132)&(S>80)&(V>50),
           (H>=15)&(H<38)&(S>100)&(V>90),(H>=38)&(H<90)&(S>85)&(V>50),
           (S<55)&(V>165),(V<85)]
    scores=sorted([(float(m.mean()),c) for m,c in zip(masks,COLORS)],reverse=True)
    top,color=scores[0]
    if top<.42 or top-scores[1][0]<.18:return 'UNKNOWN',top
    return color,top

def point(o):
    x,y,w,h=o['bbox'];return [x+w/2,y+h]

def segment_distance(p,a,b):
    p,a,b=map(np.asarray,(p,a,b));d=b-a;t=float(np.clip(np.dot(p-a,d)/max(float(d@d),1e-9),0,1))
    return float(np.linalg.norm(p-a-t*d))

def assess(objects,team,selected=None):
    players=[o for o in objects if o['kind']=='player']
    counts={c:sum(o['color']==c for o in players) for c in COLORS}
    groups=[c for c in COLORS if counts[c]>=2]
    pair=sorted(groups,key=lambda c:counts[c],reverse=True)[:2]
    teams_resolved=len(groups)==2 and sum(counts[c]>0 for c in COLORS)==2 and sum(o['color']=='UNKNOWN' for o in players)<=len(players)/2
    puck=[o for o in objects if o['kind']=='puck' and o['confidence']>=.5]
    evidence=dict(color_counts=counts,team_colors=pair if teams_resolved else [],teams_status='COLOR_HYPOTHESIS' if teams_resolved else 'UNKNOWN',
        puck_status='SINGLE_CANDIDATE' if len(puck)==1 else 'UNKNOWN',possession='UNKNOWN',roles='UNKNOWN',zones='UNKNOWN',identity='UNKNOWN',
        evaluation='INSUFFICIENT_EVIDENCE',arrows=[],selected=selected)
    en='Pause and scan. ';ru='Пауза. Осмотрим ситуацию. '
    if teams_resolved:
        en+='Two jersey-color groups are visible; team assignments still need review. '
        ru+='Видны две группы по цвету формы; принадлежность к команде ещё нужно проверить. '
    else:
        en+='I cannot reliably separate two teams in this view. '
        ru+='В этом ракурсе я не могу надёжно разделить две команды. '
    if len(puck)!=1:
        en+='The puck is uncertain. I cannot judge possession or call this a good or bad play. '
        ru+='Шайба не определена уверенно. Нельзя оценить владение или назвать это хорошим либо плохим решением. '
    elif teams_resolved:
        # No possession assertion: proximity to a detection is only a hypothesis.
        p=point(puck[0]);rank=sorted(players,key=lambda o:math.dist(point(o),p))
        carrier=rank[0] if rank else None
        clear=carrier and math.dist(point(carrier),p)<carrier['bbox'][3] and (len(rank)<2 or math.dist(point(rank[1]),p)-math.dist(point(carrier),p)>carrier['bbox'][3]*.35)
        focus=team if team!='AUTO' else (carrier['color'] if clear else None)
        if clear and carrier['color']==focus and focus in pair:
            mates=[o for o in players if o is not carrier and o['color']==focus]
            opponents=[o for o in players if o['color'] in pair and o['color']!=focus]
            choices=[]
            for mate in mates:
                a,b=point(carrier),point(mate)
                if math.dist(a,b)<carrier['bbox'][3]:continue
                # Unclassified people cannot be treated as empty ice.
                blockers=[o for o in objects if o is not carrier and o is not mate and o['kind'] in ['player','goalie','referee']]
                blocked=any(segment_distance(point(o),a,b)<o['bbox'][3]*.4 for o in blockers)
                if not blocked:choices.append(mate)
            if selected is not None:
                chosen=objects[selected]
                choices=[m for m in choices if chosen is carrier or m is chosen]
            if choices:
                mate=min(choices,key=lambda o:math.dist(point(o),point(carrier)))
                evidence['arrows']=[dict(start=point(carrier),end=point(mate),kind='CONDITIONAL_PASSING_LANE')]
                evidence['evaluation']='CONDITIONAL_SUPPORT_OPTION'
                en+='If the puck candidate and team colors are correct, the dashed arrow shows a possible support lane. No other detected person occupies this line in the image. This does not prove a safe pass: sticks, depth and off-screen players may change the decision. '
                ru+='Если шайба и цвета команд определены верно, пунктирная стрелка показывает возможную линию поддержки. На этой линии в изображении не обнаружен другой человек. Это не доказывает безопасность паса: клюшки, глубина площадки и игроки вне кадра могут изменить решение. '
            else:
                en+='A puck candidate is visible, but no clear support lane passes the image checks. This is not proof of a mistake. '
                ru+='Есть кандидат на шайбу, но проверка изображения не выявила ясную линию поддержки. Это не доказательство ошибки. '
        else:
            en+='A puck candidate is visible, but possession is uncertain. I will not assign blame or prescribe a skating route. '
            ru+='Есть кандидат на шайбу, но владение неизвестно. Я не буду приписывать ошибку или назначать маршрут движения. '
    else:
        en+='A puck candidate is visible, but team context is insufficient to evaluate the play. '
        ru+='Есть кандидат на шайбу, но данных о командах недостаточно для оценки решения. '
    en+='Coaching focus: scan before receiving, offer support and communicate. Review the moving sequence before deciding.'
    ru+='Тренерский ориентир: осмотрись до приёма, предложи поддержку и общайся с партнёрами. Перед выводом посмотри эпизод в движении.'
    evidence['narration']={'en':en,'ru':ru};return evidence

def validate_request(body,meta):
    start=body.get('start',0);end=body.get('end',min(15,meta['duration']))
    if any(type(x) not in (int,float) or not math.isfinite(x) for x in [start,end]) or not 0<=start<end<=meta['duration']+.001 or end-start>30:
        raise ValueError('Choose an interval of 0–30 seconds within the video.')
    mode=body.get('mode','team');team=body.get('team','AUTO')
    if mode not in ['team','player'] or team not in COLORS+['AUTO']:raise ValueError('Invalid coaching mode or color.')
    selection=body.get('selection')
    if mode=='player':
        from gate import validate_selections
        selection=validate_selections([selection],meta)[0]
        if not start<=meta['pts'][selection['frame_index']]<end:raise ValueError('Select a player within the requested interval.')
    return dict(start=float(start),end=float(end),mode=mode,team=team,selection=selection if mode=='player' else None)

def narrate(folder,moments,cancel):
    # Fixed local templates only. No shell, user-supplied commands or remote TTS.
    if not shutil.which('say'):return {'en':'BROWSER_FALLBACK','ru':'BROWSER_FALLBACK'}
    voices=subprocess.run(['say','-v','?'],capture_output=True,text=True,timeout=15).stdout
    chosen={}
    for lang,preferred in [('en','Samantha'),('ru','Milena')]:
        options=[line.split('  ')[0].strip() for line in voices.splitlines() if f' {lang}_' in line]
        chosen[lang]=preferred if preferred in options else (options[0] if options else None)
    state={}
    for lang,voice in chosen.items():
        if not voice:state[lang]='BROWSER_FALLBACK';continue
        state[lang]='LOCAL_AUDIO'
        for m in moments:
            if cancel.is_set():raise InterruptedError('Cancelled')
            try:
                import imageio_ffmpeg
                stem=f"voice_{m['frame_index']}_{lang}";temp=folder/(stem+'.aiff');out=folder/(stem+'.m4a')
                subprocess.run(['say','-v',voice,'-o',str(temp),m['narration'][lang]],check=True,capture_output=True,timeout=40)
                subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-i',str(temp),'-c:a','aac',str(out)],check=True,capture_output=True,timeout=30)
                temp.unlink();m.setdefault('audio',{})[lang]=out.name
            except (OSError,subprocess.SubprocessError):state[lang]='PARTIAL_BROWSER_FALLBACK'
    return state

def run_coach(video,folder,config,progress,cancel):
    import torch
    from ultralytics import YOLO
    meta=video['meta'];pts=meta['pts'];wanted=[];next_s=config['start']
    for i,t in enumerate(pts):
        if config['start']<=t<config['end'] and t>=next_s:
            wanted.append(i);next_s=t+1
    if config['mode']=='player':wanted=sorted(set(wanted+[config['selection']['frame_index']]))
    if not wanted:raise ValueError('No decoded frames in this interval. Choose a longer interval.')
    allframes=[];cap=cv2.VideoCapture(str(video['source']))
    # Per-frame detection, deliberately no unsupported persistent personnel ID claims.
    with MODEL_LOCK:
        torch.set_num_threads(4);torch.manual_seed(0);model=YOLO(str(MODEL))
        try:
            for n,i in enumerate(wanted):
                if cancel.is_set():raise InterruptedError('Cancelled')
                cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,frame=cap.read()
                if not ok:raise ValueError('Frame decode failed')
                pred=model.predict(frame,imgsz=960,conf=.25,iou=.6,classes=[2,3,4,5,6],device='cpu',verbose=False)[0]
                objects=[]
                for box,conf,cls in zip(pred.boxes.xyxy.cpu().numpy(),pred.boxes.conf.cpu().numpy(),pred.boxes.cls.cpu().numpy()):
                    x,y,x2,y2=map(float,box);b=[x,y,x2-x,y2-y];kind=model.names[int(cls)]
                    color,score=jersey_color(frame,b) if kind=='player' else ('UNKNOWN',0.)
                    objects.append(dict(kind=kind,bbox=b,confidence=float(conf),color=color,color_fraction=score))
                selection=config['selection'];selected=None
                if selection and i==selection['frame_index']:
                    from gate import box_iou
                    ranked=sorted([(box_iou(selection['bbox'],o['bbox']),j) for j,o in enumerate(objects) if o['kind']=='player'],reverse=True)
                    if ranked and ranked[0][0]>=.15 and (len(ranked)==1 or ranked[0][0]-ranked[1][0]>=.12):selected=ranked[0][1]
                evidence=assess(objects,config['team'],selected)
                if config['mode']=='player':
                    evidence['identity']='USER_SELECTED_FRAME_ONLY' if selected is not None else 'UNKNOWN'
                    if selected is None:
                        evidence['arrows']=[];evidence['evaluation']='INSUFFICIENT_PLAYER_IDENTITY'
                        evidence['narration']={ 'en':'The selected player cannot be verified in this frame. These boxes show other detection candidates, not a confirmed trajectory. I cannot judge this player’s decision. Pause on a clear view and confirm the player before individual coaching.', 'ru':'В этом кадре личность выбранного игрока не подтверждена. Рамки показывают кандидатов детектора, а не подтверждённую траекторию. Нельзя оценить решение этого игрока. Найди ясный кадр и подтверди игрока перед индивидуальным разбором.'}
                cv2.imwrite(str(folder/f'frame_{i}.jpg'),frame)
                allframes.append(dict(frame_index=i,pts_s=pts[i],objects=objects,**evidence))
                progress('coach',.8*(n+1)/len(wanted),f'Inspecting frame {n+1}/{len(wanted)}')
        finally:cap.release()
    # Five well-spaced pauses, prioritizing conditional examples without hiding missing evidence.
    ids=set(np.linspace(0,len(allframes)-1,min(5,len(allframes)),dtype=int).tolist())
    actionable=next((j for j,m in enumerate(allframes) if m['arrows']),None)
    if actionable is not None:ids.add(actionable)
    if config['selection']:ids.update(j for j,m in enumerate(allframes) if m['frame_index']==config['selection']['frame_index'])
    moments=[allframes[j] for j in sorted(ids)]
    progress('coach',.85,'Preparing bilingual narration')
    voice=narrate(folder,moments,cancel)
    if cancel.is_set():raise InterruptedError('Cancelled')
    report=dict(state='ready',message='Coaching review ready; observations require verification.',version=VERSION,config=config,
        width=meta['width'],height=meta['height'],source_sha256=meta['sha256'],model_sha256=digest(MODEL),code_sha256=CODE_SHA,
        sampling='Approximately 1 frame/second. Not continuous tracking.',sampled_frames=len(allframes),moments=moments,voice=voice,
        limitations=['Image coordinates only; no rink calibration or automatic zones.','Team colors are hypotheses, not verified team identity.','Puck detections are candidates, not confirmed possession.','No validated good/bad decision grading or tactical role recognition.','Player identity is supported only at the explicit selection frame.'],coaching_sources=[SOURCE],identity_accuracy=None)
    (folder/'coach.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (folder/'observations.json').write_text(json.dumps(allframes,ensure_ascii=False))
    with (folder/'observations.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['frame_index','pts_s','kind','x','y','w','h','confidence','color','color_fraction'])
        for m in allframes:
            for o in m['objects']:w.writerow([m['frame_index'],m['pts_s'],o['kind'],*o['bbox'],o['confidence'],o['color'],o['color_fraction']])
    return report
