"""Explicit, hash-checked retrieval of external weights. Does not execute models."""
import argparse,hashlib,json,tempfile,os
from pathlib import Path
from urllib.request import urlopen

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def fetch(asset,root,opener=urlopen):
    root=Path(root).resolve();dest=(root/asset['path']).resolve()
    if not dest.is_relative_to(root):raise ValueError('Asset path escapes checkout')
    if dest.exists():
        if sha256(dest)!=asset['sha256']:raise ValueError('Existing file has the wrong hash; preserved: '+str(dest))
        return 'already verified'
    if not asset['download_url'].startswith('https://'):raise ValueError('HTTPS required')
    dest.parent.mkdir(parents=True,exist_ok=True)
    temp=None
    try:
        with tempfile.NamedTemporaryFile(dir=dest.parent,suffix='.part',delete=False) as f:
            temp=Path(f.name)
            with opener(asset['download_url'],timeout=60) as response:
                while chunk:=response.read(1024*1024):f.write(chunk)
        if sha256(temp)!=asset['sha256']:raise ValueError('Downloaded content hash mismatch; not installed')
        # Atomic creation without overwriting a concurrent or existing file.
        os.link(temp,dest)
        return 'downloaded and verified'
    finally:
        if temp:temp.unlink(missing_ok=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--download',action='store_true',help='Download external weights after reviewing upstream licenses');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];assets=json.loads((root/'assets.json').read_text())
    if not args.download:
        for asset in assets:print(asset['path'],asset['download_url'])
        print('No download performed. Review docs/THIRD_PARTY_NOTICES.md; use --download to fetch.')
        return
    for asset in assets:print(asset['path']+': '+fetch(asset,root))
if __name__=='__main__':main()
