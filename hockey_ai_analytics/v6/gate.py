"""Manual selection binds a model hypothesis; later confirmations are explicit anchors."""
import math

POLICY = "manual_reconfirmation_v1"


def box_iou(a,b):
    x,y,w,h=a;u,v,s,t=b
    inter=max(0,min(x+w,u+s)-max(x,u))*max(0,min(y+h,v+t)-max(y,v))
    return inter/(w*h+s*t-inter) if w*h+s*t>inter else 0.


def bind(selection,objects):
    b=selection['bbox']
    ranked=sorted([(box_iou(b,[o[k] for k in ['x','y','w','h']]),o) for o in objects],key=lambda p:p[0],reverse=True)
    if not ranked or ranked[0][0]<.15:return None,'PLAYER_NOT_DETECTED_IN_SELECTION'
    if len(ranked)>1 and ranked[1][0]>=.15 and ranked[0][0]-ranked[1][0]<.12:return None,'SELECTION_CONTAINS_MULTIPLE_PLAYERS'
    return ranked[0][1],None


class ManualGate:
    def __init__(self,selections):
        self.selections={s['frame_index']:s for s in selections}
        self.first=min(self.selections);self.bound_id=None;self.seed_team=None
        self.stable=0;self.last=None;self.last_index=None;self.events=[];self.uncertain_reason=None

    def update(self,i,pts,objects):
        row=dict(frame_index=i,pts_s=pts,state='NOT_STARTED',x='',y='',w='',h='',model_track_id='',
                 reason='BEFORE_MANUAL_SELECTION',identity_status='AUTOMATIC_HYPOTHESIS',manual_anchor=False,requires_confirmation=False)
        if i<self.first:return row
        if i in self.selections:
            chosen,error=bind(self.selections[i],objects)
            self.events.append(dict(frame_index=i,pts_s=pts,selection=self.selections[i],bound_model_id=chosen['track_id'] if chosen else None,error=error))
            self.bound_id=chosen['track_id'] if chosen else None
            self.seed_team=chosen['team_hypothesis'] if chosen else None
            self.stable=3;self.last=None;self.last_index=None;self.uncertain_reason=None
            row['manual_anchor']=True
            if error:
                self.uncertain_reason=error
                row.update(state='NEEDS_SELECTION',reason=error,identity_status='UNKNOWN',requires_confirmation=True);return row
        if self.uncertain_reason:
            row.update(state='UNKNOWN',reason=self.uncertain_reason,identity_status='UNKNOWN',requires_confirmation=True)
            return row
        row.update(state='LOST',reason='SELECTED_MODEL_ID_NOT_OBSERVED',model_track_id=self.bound_id if self.bound_id is not None else '')
        candidate=next((o for o in objects if o['track_id']==self.bound_id),None)
        if candidate is None:
            self.uncertain_reason=row['reason']
            row.update(state='UNKNOWN',identity_status='UNKNOWN',requires_confirmation=True,model_track_id='')
            return row
        c=candidate;pc=(c['x']+c['w']/2,c['y']+c['h']/2)
        reason=None
        # Colour is a weak consistency check. Manual selection can bind any colour.
        if self.seed_team=='RED' and c['team_hypothesis']!='RED':reason='APPEARANCE_UNCERTAIN'
        for other in objects:
            if other['track_id']==self.bound_id:continue
            oc=(other['x']+other['w']/2,other['y']+other['h']/2)
            if c['team_hypothesis']==other['team_hypothesis'] and math.dist(pc,oc)<.65*c['h']:
                reason=reason or 'SIMILAR_PLAYER_NEARBY'
            a=[c[k] for k in ['x','y','w','h']];b=[other[k] for k in ['x','y','w','h']]
            inter=max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))*max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
            if other['detector_conf']>=.25 and inter/min(a[2]*a[3],b[2]*b[3])>.15:reason=reason or 'OVERLAP_REQUIRES_REVIEW'
        if self.last is not None and math.dist(pc,self.last)>20*max(1,i-self.last_index):reason=reason or 'MOTION_UNCERTAIN'
        if reason:
            self.uncertain_reason=reason
            row.update(state='UNKNOWN',reason=reason,identity_status='UNKNOWN',requires_confirmation=True,model_track_id='');return row
        row.update(state='TRACKING',reason='SAME_BOUND_MODEL_ID',**{k:c[k] for k in ['x','y','w','h']})
        if row['manual_anchor']:row['identity_status']='USER_SELECTED_START'
        self.last=pc;self.last_index=i
        return row


def validate_selections(items,meta):
    if not isinstance(items,list) or not 1<=len(items)<=100:raise ValueError('Выделите одного игрока на кадре.')
    result=[]
    for s in items:
        if not isinstance(s,dict):raise ValueError('Неверное подтверждение игрока.')
        f=s.get('frame_index');b=s.get('bbox')
        if type(f) is not int or not 0<=f<meta['frame_count']:raise ValueError('Выбранный кадр отсутствует в видео.')
        if not isinstance(b,list) or len(b)!=4 or not all(type(v) in (int,float) and math.isfinite(v) for v in b):raise ValueError('Неверная рамка игрока.')
        x,y,w,h=b
        if w<3 or h<6 or x<0 or y<0 or x+w>meta['width']+.01 or y+h>meta['height']+.01:raise ValueError('Рамка должна охватывать игрока внутри кадра.')
        if result and s.get('confirmed_same_player') is not True:raise ValueError('Подтвердите, что повторно выбран тот же игрок.')
        result.append(dict(frame_index=f,bbox=list(map(float,b)),confirmed_same_player=bool(s.get('confirmed_same_player'))))
    if [s['frame_index'] for s in result]!=sorted(set(s['frame_index'] for s in result)):raise ValueError('Повторные подтверждения должны идти по порядку кадров.')
    return result
