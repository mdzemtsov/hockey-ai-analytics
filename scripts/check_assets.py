"""Validate separately obtained models without downloading or executing them."""
import hashlib,json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
failed=False
for asset in json.loads((root/'assets.json').read_text()):
    p=root/asset['path']
    if not p.is_file():
        print('MISSING:',asset['path'],'—',asset['source']);failed=True;continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    ok=h.hexdigest()==asset['sha256'];print('OK:' if ok else 'HASH MISMATCH:',asset['path']);failed|=not ok
sys.exit(int(failed))
