"""Conservative output gate. A model ID alone is never an identity certificate."""
import numpy as np


def center(o):return np.array([o['x']+o['w']/2,o['y']+o['h']/2])


def overlap(a,b):
    inter=max(0,min(a['x']+a['w'],b['x']+b['w'])-max(a['x'],b['x']))*max(0,min(a['y']+a['h'],b['y']+b['h'])-max(a['y'],b['y']))
    return inter/min(a['w']*a['h'],b['w']*b['h'])


class TargetGate:
    def __init__(self,seed_frame=90,seed_point=(195.,302.)):
        self.seed_frame=seed_frame;self.seed_point=seed_point
        self.bound_id=None;self.stable=0;self.last_center=None;self.last_index=None

    def update(self,i,pts,objects):
        row=dict(frame_index=i,pts_s=pts,state='NOT_STARTED',x='',y='',w='',h='',track_id='',reason='BEFORE_SEED',identity_status='NOT_CERTIFIED')
        if i<self.seed_frame:return row
        if i==self.seed_frame:
            x,y=self.seed_point
            inside=[o for o in objects if o['x']<=x<=o['x']+o['w'] and o['y']<=y<=o['y']+o['h'] and o['team_hypothesis']=='RED']
            if len(inside)==1:self.bound_id=inside[0]['track_id'];self.stable=3
        row.update(state='LOST',reason='TARGET_ID_NOT_OBSERVED',track_id=self.bound_id if self.bound_id is not None else '')
        candidates=[o for o in objects if o['track_id']==self.bound_id]
        if len(candidates)!=1:self.stable=0;return row
        o=candidates[0];pc=center(o);others=[r for r in objects if r['track_id']!=self.bound_id]
        nearby=[r for r in others if r['team_hypothesis']=='RED' and np.linalg.norm(pc-center(r))<.65*o['h']]
        overlapping=[r for r in others if r.get('detector_conf',1.)>=.25 and overlap(o,r)>.15]
        reason=None
        if o['team_hypothesis']!='RED':reason='TEAM_APPEARANCE_UNCERTAIN'
        elif nearby:reason='NEARBY_SIMILAR_TEAM_IDENTITY_UNCERTAIN'
        elif overlapping:reason='OVERLAPPING_PERSON_IDENTITY_UNCERTAIN'
        elif self.last_center is not None and np.linalg.norm(pc-self.last_center)>20*max(1,i-self.last_index):reason='EXCESSIVE_DISPLACEMENT'
        if reason:
            self.stable=0;row.update(state='AMBIGUOUS',reason=reason);return row
        self.stable+=1
        if self.stable<3:
            row.update(state='AMBIGUOUS',reason='RECOVERY_CONFIRMATION_PENDING');return row
        row.update(state='TRACKING',reason='SAME_BOUND_MODEL_ID_UNIQUE_UNOBSCURED_CANDIDATE',**{k:o[k] for k in ['x','y','w','h']})
        self.last_center=pc;self.last_index=i
        return row
