from pathlib import Path
import hashlib,json,gdown
ROOT=Path(__file__).resolve().parent/'models';ROOT.mkdir(exist_ok=True)
items=[('hockey_parseq.ckpt','1FyM31xvSXFRusN0sZH0EWXoHwDfB9WIE','5a31899866c092ff26b245898d1fb16ecaabcd091afaadeb6a9c056ca6c846e3'),('hockey_legibility.pth','1RfxINtZ_wCNVF8iZsiMYuFOP7KMgqgDp','98de82bc56fe2943bf9db961325fbd08a4508532182b3c27f8d4f4e3786bc786')]
for name,ident,sha in items:
 p=ROOT/name
 if not p.exists():gdown.download(id=ident,output=str(p),quiet=False)
 actual=hashlib.sha256(p.read_bytes()).hexdigest()
 if actual!=sha:raise ValueError(f'Published model changed: {name}')
 print(name,actual)
