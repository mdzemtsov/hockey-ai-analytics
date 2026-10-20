"""Conservative experimental V4. No GT, manual corrections or re-ID after seed.

Heuristic torso proposals + local foreground structure and FB optical-flow gate.
This is not a learned hockey-player detector or an identity recognition model.
"""
import cv2
import numpy as np

CONFIG = dict(seed_frame=90, seed_bbox=[177,276,36,52],
              min_component_area=12, max_component_area=600,
              motion_gate_px=18, ambiguity_distance_px=10,
              max_fb_error_px=1.5, min_flow_points=4,
              max_flow_spread_px=3.5, max_speed_px_frame=12)


def red_mask(frame):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=(((hsv[:,:,0]<12)|(hsv[:,:,0]>168)) & (hsv[:,:,1]>90) & (hsv[:,:,2]>45)).astype('uint8')*255
    return cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))


def components(frame):
    mask=red_mask(frame)
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
    out=[]
    for j in range(1,n):
        x,y,w,h,area=stats[j]
        if not CONFIG['min_component_area']<=area<=CONFIG['max_component_area']:continue
        if not (4<=w<=45 and 5<=h<=40 and y>90):continue
        # Require neutral/dark structure above and below the red torso, locally.
        patch=frame[max(0,y-8):min(464,y+h+16),max(0,x-3):min(848,x+w+3)]
        gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
        if (gray<130).sum()<8:continue
        out.append(dict(torso=np.array(centers[j]),xywh=[int(x),int(y),int(w),int(h)],area=int(area)))
    return out,mask


def select(candidates,prediction):
    ranked=sorted([(float(np.linalg.norm(c['torso']-prediction)),c) for c in candidates],key=lambda z:z[0])
    ranked=[r for r in ranked if r[0]<=CONFIG['motion_gate_px']]
    if not ranked:return None,'NO_TORSO_IN_MOTION_GATE'
    if len(ranked)>1 and ranked[1][0]-ranked[0][0]<CONFIG['ambiguity_distance_px']:
        return None,'AMBIGUOUS_RED_CANDIDATES'
    return ranked[0][1],'UNIQUE_LOCAL_CANDIDATE'


def run(frames,pts):
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    rows=[];last_gray=None;points=None;torso=None;lost=False;box=None
    for i,frame in enumerate(frames):
        row=dict(frame_index=i,pts_s=pts[i],state='NOT_STARTED',x='',y='',w='',h='',observed_id='',reason='BEFORE_SEED')
        if i<CONFIG['seed_frame']:rows.append(row);continue
        if lost:
            row.update(state='LOST',reason='LATCHED_LOST_NO_AUTOMATIC_REID');rows.append(row);continue
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cs,mask=components(frame)
        if i==CONFIG['seed_frame']:
            x,y,w,h=CONFIG['seed_bbox'];box=[x,y,w,h]
            candidate,reason=select(cs,np.array([x+w/2,y+h*.40]))
        else:
            candidate=None;reason='INSUFFICIENT_FLOW_POINTS'
            if points is not None and len(points)>=CONFIG['min_flow_points']:
                nxt,st,_=cv2.calcOpticalFlowPyrLK(last_gray,gray,points,None,winSize=(15,15),maxLevel=2)
                back,sb,_=cv2.calcOpticalFlowPyrLK(gray,last_gray,nxt,None,winSize=(15,15),maxLevel=2)
                fb=np.linalg.norm(back-points,axis=2).ravel()
                good=(st.ravel()==1)&(sb.ravel()==1)&(fb<CONFIG['max_fb_error_px'])
                if good.sum()>=CONFIG['min_flow_points']:
                    delta=(nxt-points).reshape(-1,2)[good];motion=np.median(delta,axis=0)
                    spread=float(np.median(np.linalg.norm(delta-motion,axis=1)))
                    if spread<=CONFIG['max_flow_spread_px'] and np.linalg.norm(motion)<=CONFIG['max_speed_px_frame']:
                        candidate,reason=select(cs,torso+motion)
                    else:reason='INCONSISTENT_OR_EXCESSIVE_FLOW'
        if candidate is None:
            lost=True;row.update(state='AMBIGUOUS' if reason.startswith('AMBIGUOUS') else 'LOST',reason=reason)
            rows.append(row);continue
        torso=candidate['torso'];x,y,w,h=candidate['xywh']
        # Box estimate for display only; torso proposals do not measure full body.
        bw=max(18,min(44,w*1.8));bh=max(30,min(60,h*2.2))
        box=[float(torso[0]-bw/2),float(torso[1]+5-bh/2),float(bw),float(bh)]
        row.update(state='TRACKING',x=box[0],y=box[1],w=box[2],h=box[3],reason=reason)
        # Reset features only on accepted torso; never train on an ambiguous frame.
        local=np.zeros_like(mask);pad=4
        local[max(0,y-pad):y+h+pad,max(0,x-pad):x+w+pad]=255
        points=cv2.goodFeaturesToTrack(gray,maxCorners=40,qualityLevel=.01,minDistance=2,mask=local,blockSize=3)
        last_gray=gray
        rows.append(row)
    return rows
