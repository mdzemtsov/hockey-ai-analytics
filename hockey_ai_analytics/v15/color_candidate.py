"""Frozen experimental jersey appearance classifier, not personnel identity."""
import cv2,numpy as np
COLORS=['RED','BLUE','YELLOW','GREEN','WHITE','DARK']
def classify(frame,box):
 x,y,w,h=box;H,W=frame.shape[:2]
 crop=frame[max(0,int(y+.18*h)):min(H,int(y+.58*h)),max(0,int(x+.15*w)):min(W,int(x+.85*w))]
 if crop.size<60:return 'UNKNOWN',0.
 hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV);a,s,v=cv2.split(hsv)
 masks=[((a<12)|(a>168))&(s>95)&(v>65),(a>95)&(a<132)&(s>80)&(v>50),(a>=15)&(a<38)&(s>100)&(v>90),(a>=38)&(a<90)&(s>85)&(v>50),(s<55)&(v>165),(v<85)]
 scores=[float(m.mean()) for m in masks];chrom=max(range(4),key=lambda i:scores[i])
 # Jersey numbers and stripes can occupy the middle of a colored uniform.
 if scores[chrom]>=.24 and scores[4]<.55:return COLORS[chrom],scores[chrom]
 if scores[4]>=.38 and scores[4]-max(scores[:4])>=.12:return 'WHITE',scores[4]
 if scores[5]>=.42:return 'DARK',scores[5]
 return 'UNKNOWN',max(scores)
