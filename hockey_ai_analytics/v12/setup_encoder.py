from pathlib import Path
import urllib.request,json,torch,hashlib
r=Path(__file__).resolve().parent
torch.set_num_threads(4);torch.hub.set_dir(str(r/'hub'))
meta=json.load(urllib.request.urlopen('https://api.github.com/repos/facebookresearch/dinov2/commits/main',timeout=30));sha=meta['sha']
model=torch.hub.load('facebookresearch/dinov2:'+sha,'dinov2_vits14',trust_repo=True).eval()
p=r/'encoder_source.json'
p.write_text(json.dumps(dict(repo='https://github.com/facebookresearch/dinov2',commit=sha,model='dinov2_vits14',purpose='General image descriptor; not a validated hockey identity recognizer',weights=[dict(path=str(x.relative_to(r)),sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in (r/'hub/checkpoints').glob('*')]),indent=2))
with torch.inference_mode():print(model(torch.zeros(1,3,224,112)).shape)
