from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request,json,hashlib,time
R=Path(__file__).resolve().parent
for x in json.loads((R/'candidate_lock.json').read_text())['files']:
 assert hashlib.sha256((R.parent/x['path']).read_bytes()).hexdigest()==x['sha256']
p=json.loads((R/'protocol.json').read_text());seq=p['test_match'];first,last=p['test_query_frames']
out=R/'test/img1';out.mkdir(parents=True,exist_ok=True)
def fetch(n):
 path=out/f'{n:06d}.jpg';url=f'https://raw.githubusercontent.com/harshap-ai/VIP-HTD/main/personnel-level-format/test/{seq}/img1/{n:06d}.jpg'
 if not path.exists():
  for attempt in range(4):
   try:
    data=urllib.request.urlopen(url,timeout=45).read();assert data[:2]==b'\xff\xd8';path.write_bytes(data);break
   except Exception:
    if attempt==3:raise
    time.sleep(1)
 return dict(frame=n,url=url,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
with ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(fetch,range(first,last+1)))
(R/'test/source_manifest.json').write_text(json.dumps(rows,indent=2));print('Downloaded',len(rows),'source images')
