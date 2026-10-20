import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));from consensus import ConsensusGate
class GuardTests(unittest.TestCase):
 def d(self,ident='1',team='LIGHT'):return dict(id=ident,team_hypothesis=team,cls=4,x1=0,y1=0,x2=10,y2=10)
 def test_requires_three(self):
  g=ConsensusGate('LIGHT');self.assertIsNone(g.update([0,0,10,10],[self.d()]));self.assertIsNone(g.update([0,0,10,10],[self.d()]));self.assertIsNotNone(g.update([0,0,10,10],[self.d()]))
 def test_ambiguous_duplicates(self):
  g=ConsensusGate('LIGHT')
  for _ in range(4):self.assertIsNone(g.update([0,0,10,10],[self.d(),self.d('2')]))
 def test_other_team(self):
  g=ConsensusGate('LIGHT')
  for _ in range(4):self.assertIsNone(g.update([0,0,10,10],[self.d(team='DARK')]))
 def test_gap_resets(self):
  g=ConsensusGate('LIGHT')
  for _ in range(3):g.update([0,0,10,10],[self.d()])
  self.assertIsNone(g.update(None,[]));self.assertIsNone(g.update([0,0,10,10],[self.d()]))
 def test_changed_identity_requires_stability(self):
  g=ConsensusGate('LIGHT')
  for _ in range(3):g.update([0,0,10,10],[self.d()])
  self.assertIsNone(g.update([0,0,10,10],[self.d('2')]))
if __name__=='__main__':unittest.main()
