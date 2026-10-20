from pathlib import Path
import urllib.request,json,hashlib,concurrent.futures,time
root=Path(__file__).resolve().parent/'data';root.mkdir(exist_ok=True)
base='https://raw.githubusercontent.com/harshap-ai/VIP-HTD/main/'
items={
'PIT_VS_SJ_001.mp4':base+'clips/PIT_VS_SJ/001.mp4',
'personnel_gt.txt':base+'personnel-level-format/test/PIT_VS_SJ_001/gt/gt.txt',
'seqinfo.ini':base+'personnel-level-format/test/PIT_VS_SJ_001/seqinfo.ini',
'VIP_HTD_LICENSE':base+'LICENSE',
'VIP_HTD_README.md':base+'readme.md',
'HockeyAI.pt':'https://huggingface.co/SimulaMet-HOST/HockeyAI/resolve/main/HockeyAI_model_weight.pt',
'HockeyAI_model_README.md':'https://huggingface.co/SimulaMet-HOST/HockeyAI/raw/main/README.md',
'HockeyAI_Dataset.zip':'https://huggingface.co/datasets/SimulaMet-HOST/HockeyAI/resolve/main/HockeyAI_Dataset.zip',
'HockeyAI_dataset_README.md':'https://huggingface.co/datasets/SimulaMet-HOST/HockeyAI/raw/main/README.md'}
def get(item):
 name,url=item;p=root/name
 if not p.exists():
  with urllib.request.urlopen(url,timeout=60) as r,p.with_suffix(p.suffix+'.part').open('wb') as f:
   while data:=r.read(1024*1024):f.write(data)
  p.with_suffix(p.suffix+'.part').replace(p)
 h=hashlib.sha256()
 with p.open('rb') as f:
  while data:=f.read(1024*1024):h.update(data)
 result=dict(file=name,url=url,bytes=p.stat().st_size,sha256=h.hexdigest());print(name,result['bytes'],flush=True);return result
with concurrent.futures.ThreadPoolExecutor(3) as ex:results=list(ex.map(get,items.items()))
(root/'download_manifest.json').write_text(json.dumps(results,indent=2)+'\n')
