"""V9.1: team hypotheses may not veto agreement with the originally selected track ID."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'v8'))
from consensus import ConsensusGate
class RevisedGate:
 def __init__(self,team,initial_id):
  self.team=team if team in ('LIGHT','DARK') else 'LIGHT';self.known_team=team in ('LIGHT','DARK');self.initial_id=initial_id;self.gate=ConsensusGate(self.team)
 def update(self,sam_box,detections):
  eligible=[]
  for d in detections:
   if d['id']==self.initial_id:
    eligible.append({**d,'team_hypothesis':self.team})
   elif self.known_team:eligible.append(d)
  return self.gate.update(sam_box,eligible)
