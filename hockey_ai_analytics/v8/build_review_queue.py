"""Create compact manual-review intervals from tracker loss; no automatic identity repair."""
from pathlib import Path
import csv,json
r=Path(__file__).resolve().parent;source=r.parent/'v7/results/hockey/selected.csv';rows=list(csv.DictReader(source.open()));intervals=[]
for row in rows:
 if row['state']!='lost':continue
 f=int(row['frame'])
 if intervals and intervals[-1]['last_frame']==f-1:intervals[-1]['last_frame']=f
 else:intervals.append(dict(first_frame=f,last_frame=f,reason='SELECTED_ID_NOT_OBSERVED',action='Review exit/occlusion and confirm reappearance; do not auto-rebind'))
for a in intervals:a.update(start_s=(a['first_frame']-1)/30,end_s=a['last_frame']/30)
(r/'review_queue.json').write_text(json.dumps(intervals,indent=2));print(intervals)
