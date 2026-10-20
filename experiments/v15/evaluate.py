import json,csv,sys,hashlib
from pathlib import Path
import cv2
R=Path(__file__).resolve().parent;BASE=R.parents[1]/'hockey_ai_analytics';sys.path.insert(0,str(BASE/'v6'))
from coach import jersey_color
from color_candidate import classify
rows=json.loads((R/'manifest.json').read_text());labels=json.loads((R/'labels.json').read_text())
summary={};out=[]
for r in rows:
 if str(r['id']) not in labels:continue
 truth=labels[str(r['id'])];im=cv2.imread(str(BASE/'v6/work/jobs'/r['job']/f"frame_{r['frame']}.jpg"))
 base=jersey_color(im,r['box'])[0];new=classify(im,r['box'])[0];out.append(dict(**r,truth=truth,old=base,new=new))
for split in ['development','validation']:
 for method in ['old','new']:
  eligible=[r for r in out if r['split']==split and r['truth'] not in ['MIXED','REFEREE']];rejected=[r for r in out if r['split']==split and r['truth'] in ['MIXED','REFEREE']]
  summary[split+'_'+method]=dict(n=len(eligible),correct=sum(r[method]==r['truth'] for r in eligible),wrong=sum(r[method] not in [r['truth'],'UNKNOWN'] for r in eligible),unknown=sum(r[method]=='UNKNOWN' for r in eligible),mixed_or_referee_assigned=sum(r[method]!='UNKNOWN' for r in rejected),excluded_count=len(rejected))
(R/'comparison.json').write_text(json.dumps(out,indent=2));(R/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
