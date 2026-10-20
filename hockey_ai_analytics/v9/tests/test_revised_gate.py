import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));from revised_gate import RevisedGate
class RevisedTests(unittest.TestCase):
 def d(self,ident='1',team='DARK'):return dict(id=ident,team_hypothesis=team,cls=4,x1=0,y1=0,x2=10,y2=10)
 def test_original_id_team_veto_relaxed_with_agreement(self):
  g=RevisedGate('LIGHT','1');out=None
  for _ in range(3):out=g.update([0,0,10,10],[self.d()])
  self.assertIsNotNone(out)
 def test_new_id_other_team_still_blocked(self):
  g=RevisedGate('LIGHT','1')
  for _ in range(4):self.assertIsNone(g.update([0,0,10,10],[self.d('2')]))
 def test_sam_disagreement_cannot_be_bypassed(self):
  g=RevisedGate('LIGHT','1')
  for _ in range(4):self.assertIsNone(g.update([30,30,40,40],[self.d()]))
 def test_unknown_team_only_allows_original_id(self):
  g=RevisedGate('UNKNOWN','1')
  for _ in range(4):self.assertIsNone(g.update([0,0,10,10],[self.d('2','LIGHT')]))
if __name__=='__main__':unittest.main()
