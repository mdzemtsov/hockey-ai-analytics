from pathlib import Path
import hashlib,json,datetime
r=Path(__file__).resolve().parent
paths=[r/'recovery.py',r/'recovery_config.json',r/'run_recovery.py',r.parent/'v10/appearance_tracker.py',r.parent/'v5/botsort_reid.yaml',r.parent/'v7/data/HockeyAI.pt']
p=r/'candidate_lock.json'
assert not p.exists(), 'Do not overwrite a lock'
p.write_text(json.dumps(dict(locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidate='causal_recovery_v1',validation='CNTRL_VS_PAC_001 frames 901–1500; fresh tracker at 901; baseline and candidate share predictions; manual seeds from frame901 before reference inspection',files=[dict(path=str(x.relative_to(r.parent)),sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in paths]),indent=2))
print(p)
