from pathlib import Path
import urllib.request,json,hashlib,concurrent.futures,time
ROOT=Path(__file__).resolve().parent
SEQUENCES=['CAR_VS_BOS_001','CGY_VS_DAL_003','CHI_VS_TOR_004']
items=[]
for seq in SEQUENCES:
 for suffix in ['seqinfo.ini','gt/gt.txt']+[f'img1/{i:06d}.jpg' for i in range(1,301)]:items.append((seq,suffix))
def get(item):
 seq,suffix=item;url=f'https://raw.githubusercontent.com/harshap-ai/VIP-HTD/main/personnel-level-format/test/{seq}/{suffix}';p=ROOT/'data'/seq/suffix;p.parent.mkdir(parents=True,exist_ok=True)
 for attempt in range(4):
  try:
   if not p.exists():
    data=urllib.request.urlopen(url,timeout=60).read();p.write_bytes(data)
   return dict(path=str(p.relative_to(ROOT)),url=url,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
  except Exception:
   if attempt==3:raise
   time.sleep(1)
with concurrent.futures.ThreadPoolExecutor(10) as pool:rows=list(pool.map(get,items))
(ROOT/'source_manifest.json').write_text(json.dumps(rows,indent=2));print('Downloaded 900 frames and annotations for three matches')
