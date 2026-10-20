"""Independent local motion check. It cannot certify identity or reacquire after absence."""
from pathlib import Path
import sys,csv,json,argparse
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));from run_match import evaluate_selected,write_csv

def check(prev,cur,old_box,new_box):
 if old_box is None or new_box is None:return dict(status='UNKNOWN',reason='NO_CONTIGUOUS_BOX',points=0,relative_error=None)
 x1,y1,x2,y2=old_box;w=x2-x1;h=y2-y1;mask=np.zeros(prev.shape,np.uint8)
 a=max(0,int(x1+.2*w));b=min(prev.shape[1],int(x2-.2*w));c=max(0,int(y1+.15*h));d=min(prev.shape[0],int(y1+.7*h));mask[c:d,a:b]=255
 points=cv2.goodFeaturesToTrack(prev,maxCorners=40,qualityLevel=.01,minDistance=3,mask=mask)
 if points is None or len(points)<5:return dict(status='UNKNOWN',reason='TOO_FEW_FEATURES',points=0,relative_error=None)
 p1,st,_=cv2.calcOpticalFlowPyrLK(prev,cur,points,None,winSize=(21,21),maxLevel=3);back,st2,_=cv2.calcOpticalFlowPyrLK(cur,prev,p1,None,winSize=(21,21),maxLevel=3)
 ok=(st.reshape(-1)==1)&(st2.reshape(-1)==1)&(np.linalg.norm(back-points,axis=2).reshape(-1)<1.5)
 if ok.sum()<5:return dict(status='UNKNOWN',reason='FORWARD_BACKWARD_DISAGREEMENT',points=int(ok.sum()),relative_error=None)
 flow=(p1-points).reshape(-1,2)[ok];delta=np.median(flow,axis=0);center=np.array([(x1+x2)/2,(y1+y2)/2])+delta;newcenter=np.array([(new_box[0]+new_box[2])/2,(new_box[1]+new_box[3])/2]);error=float(np.linalg.norm(newcenter-center)/max(h,1));status='AGREES' if error<=.35 else 'DISAGREES'
 return dict(status=status,reason='LOCAL_FLOW_VS_BOX_MOTION',points=int(ok.sum()),relative_error=error)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('sequence');a=ap.parse_args();seq=a.sequence;out=ROOT/'results'/seq;data=ROOT/'data'/seq;rows=list(csv.DictReader((out/'sam.csv').open()));hy=list(csv.DictReader((out/'hybrid.csv').open()));previous=None;old=None;records=[];filtered=[]
 for r,hyp in zip(rows,hy):
  f=int(r['frame']);im=cv2.imread(str(data/'img1'/f'{f:06d}.jpg'),0);H,W=im.shape;hh=round(H*848/W);im=cv2.resize(im,(848,hh));native=json.loads(r['box']);box=(np.array(native)*np.array([848/W,hh/H]*2)).tolist() if native else None
  result=check(previous,im,old,box) if previous is not None else dict(status='UNKNOWN',reason='FIRST_FRAME',points=0,relative_error=None);records.append(dict(frame=f,**result));hb=json.loads(hyp['box']);filtered.append(None if result['status']=='DISAGREES' else hb);previous=im;old=box
 write_csv(out/'motion_checks.csv',records);gt=np.loadtxt(data/'gt/gt.txt',delimiter=',');target=json.loads((ROOT/'selections.json').read_text())[seq]['published_id'];scored,metrics=evaluate_selected(filtered,gt,target);write_csv(out/'hybrid_motion.csv',scored);metrics['motion_status_counts']={s:sum(r['status']==s for r in records) for s in ['AGREES','DISAGREES','UNKNOWN']};metrics['policy']='Reject explicit disagreement only; UNKNOWN does not certify motion';(out/'motion_metrics.json').write_text(json.dumps(metrics,indent=2));print(seq,metrics)
if __name__=='__main__':main()
