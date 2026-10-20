from pathlib import Path
import urllib.request,concurrent.futures,hashlib,json,time
root=Path(__file__).resolve().parent/'data/frames'
root.mkdir(parents=True,exist_ok=True)
def get(i):
 p=root/f'{i:06d}.jpg';url=f'https://raw.githubusercontent.com/harshap-ai/VIP-HTD/main/personnel-level-format/test/PIT_VS_SJ_001/img1/{p.name}'
 for attempt in range(4):
  try:
   if not p.exists():p.write_bytes(urllib.request.urlopen(url,timeout=60).read())
   return dict(frame=i,url=url,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
  except Exception:
   if attempt==3:raise
   time.sleep(1)
with concurrent.futures.ThreadPoolExecutor(8) as ex: rows=list(ex.map(get,range(1,301)))
(root.parent/'frames_manifest.json').write_text(json.dumps(rows,indent=2));print('300 annotated source frames downloaded')
