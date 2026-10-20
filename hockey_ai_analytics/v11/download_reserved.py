from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request,hashlib,json,time
r=Path(__file__).resolve().parent
assert (r/'candidate_lock.json').exists()
out=r/'reserved/img1';out.mkdir(parents=True,exist_ok=True)
def fetch(n):
 p=out/f'{n:06d}.jpg'
 url=f'https://raw.githubusercontent.com/harshap-ai/VIP-HTD/main/personnel-level-format/test/CNTRL_VS_PAC_001/img1/{n:06d}.jpg'
 if not p.exists():
  for attempt in range(4):
   try:
    data=urllib.request.urlopen(url,timeout=45).read()
    assert data[:2]==b'\xff\xd8'
    p.write_bytes(data);break
   except Exception:
    if attempt==3:raise
    time.sleep(1)
 return dict(frame=n,url=url,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
with ThreadPoolExecutor(max_workers=8) as pool:
 rows=[]
 for row in pool.map(fetch,range(901,1501)):
  rows.append(row)
  if len(rows)%100==0:print(len(rows),flush=True)
(r/'reserved/source_manifest.json').write_text(json.dumps(rows,indent=2))
