from pathlib import Path
import json,hashlib,zipfile
ROOT=Path(__file__).resolve().parent
external=["v7/benchmark.py","v7/data/HockeyAI.pt","v8/models/sam2.1_t.pt",
          "v5/botsort_reid.yaml","v9/run_match.py","v9/selections.json",
          "v7/team_analysis.py","v8/consensus.py","v6/gate.py"]
def record(p,root):
    return dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(ROOT/"external_dependencies.json").write_text(json.dumps([record(ROOT.parent/p,ROOT.parent) for p in external],indent=2))
files=[]
for p in sorted(ROOT.rglob("*")):
    if not p.is_file() or any(x in p.relative_to(ROOT).parts for x in ("data","__pycache__")):continue
    if p.name in ("input.mp4","artifact_manifest.json"):continue
    if p.suffix not in (".py",".json",".csv",".md",".mp4",".jpg",".yaml",".txt",".log"):continue
    files.append(p)
(ROOT/"artifact_manifest.json").write_text(json.dumps([record(p,ROOT) for p in files],indent=2))
bundle=ROOT.parent/"pilot_001k_v10_night_results.zip"
with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as z:
    for p in files+[ROOT/"artifact_manifest.json"]:z.write(p,str(p.relative_to(ROOT.parent)))
with zipfile.ZipFile(bundle) as z:assert z.testzip() is None
print(json.dumps(dict(bundle=str(bundle.resolve()),bytes=bundle.stat().st_size,files=len(files)+1,
                     sha256=hashlib.sha256(bundle.read_bytes()).hexdigest())))
