from pathlib import Path
import sys,json,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evidence import EvidenceGate,GuardedRecovery,Recovery
R=Path(__file__).resolve().parents[1]
class EvidenceTests(unittest.TestCase):
 def setUp(self):
  self.cfg=json.loads((R/'config.json').read_text())
  self.a=np.array([1.,0.]);self.b=np.array([0.,1.])
  self.gate=EvidenceGate(self.cfg,[self.a,self.b],0,self.a,[self.b])
 def row(self,team=None,embedding=None):
  return dict(team_feature=self.a if team is None else team,embedding=self.a if embedding is None else embedding)
 def test_correct_separated_evidence(self):self.assertTrue(self.gate.check(self.row())[0])
 def test_opponent_rejected_even_with_matching_embedding(self):self.assertFalse(self.gate.check(self.row(team=self.b))[0])
 def test_similar_teammate_rejected(self):self.assertFalse(self.gate.check(self.row(embedding=self.b))[0])
 def test_tied_identity_rejected(self):self.assertFalse(self.gate.check(self.row(embedding=np.array([.707,.707])))[0])
 def test_uncertain_team_rejected(self):self.assertFalse(self.gate.check(self.row(team=np.array([.5,.5])))[0])
 def test_no_embedding_abstains(self):
  r=self.row();r['embedding']=None;self.assertFalse(self.gate.check(r)[0])
 def test_same_team_examples_rejected(self):
  g=EvidenceGate(self.cfg,[self.a,self.a],0,self.a,[]);self.assertFalse(g.check(self.row())[0])
 def test_ablation_omits_identity_only(self):
  g=EvidenceGate(self.cfg,[self.a,self.b],0,self.a,[self.b],False)
  self.assertTrue(g.check(self.row(embedding=self.b))[0]);self.assertFalse(g.check(self.row(team=self.b))[0])
 def test_same_track_id_does_not_bypass_gate(self):
  cfg=json.loads((R.parent/'v11/recovery_config.json').read_text())
  b=Recovery(cfg,[0,0,40,100],self.a,30);g=GuardedRecovery(b,self.gate)
  r=dict(id=1,cls=4,box=[0,0,40,100],feature=self.a,**self.row())
  self.assertIsNotNone(g.update([r])['box'])
  r['team_feature']=self.b
  self.assertIsNone(g.update([r])['box'])
 def test_rejected_opponent_still_counts_as_occlusion(self):
  cfg=json.loads((R.parent/'v11/recovery_config.json').read_text())
  g=GuardedRecovery(Recovery(cfg,[0,0,40,100],self.a,30),self.gate)
  r=dict(id=1,cls=4,box=[0,0,40,100],feature=self.a,**self.row())
  g.update([r])
  opponent=dict(id=2,cls=4,box=[10,0,50,100],feature=self.b,**self.row(team=self.b))
  self.assertEqual(g.update([r,opponent])['state'],'OCCLUDED')
if __name__=='__main__':unittest.main()
