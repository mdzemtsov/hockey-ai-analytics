from pathlib import Path
import json, urllib.request, hashlib, concurrent.futures, time,sys
ROOT = Path(__file__).resolve().parent
protocol=sys.argv[1] if len(sys.argv)>1 else "long_holdout_protocol.json"
cfg = json.loads((ROOT/protocol).read_text())
seq = cfg["sequence"]
suffixes = ["seqinfo.ini", "gt/gt.txt"] + [f"img1/{i:06d}.jpg" for i in range(cfg["first_frame"],cfg["last_frame"]+1)]


def get(suffix):
    url = f"https://raw.githubusercontent.com/harshap-ai/VIP-HTD/main/{cfg['split']}/{seq}/{suffix}"
    path = ROOT/"data"/seq/suffix
    path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(4):
        try:
            if not path.exists():
                data = urllib.request.urlopen(url,timeout=45).read()
                temp = path.with_suffix(path.suffix+".partial")
                temp.write_bytes(data)
                temp.replace(path)
            return dict(path=str(path.relative_to(ROOT)),url=url,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1)


with concurrent.futures.ThreadPoolExecutor(8) as pool:
    rows=[]
    for row in pool.map(get,suffixes):
        rows.append(row)
        if len(rows)%200==0:
            print("Downloaded",len(rows),flush=True)
manifest="long_holdout_sources.json" if protocol=="long_holdout_protocol.json" else Path(protocol).stem+"_sources.json"
(ROOT/manifest).write_text(json.dumps(rows,indent=2))
print("Complete",len(rows),flush=True)
