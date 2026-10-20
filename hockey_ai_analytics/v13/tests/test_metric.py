from pathlib import Path
import sys,json,unittest
import numpy as np,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from metric_model import MetricHead,objective
from retrieval import rank,summarize
class MetricTests(unittest.TestCase):
 def test_normalized_embeddings(self):
  z=MetricHead()(torch.randn(3,384));self.assertEqual(tuple(z.shape),(3,128));self.assertTrue(torch.allclose(z.norm(dim=1),torch.ones(3),atol=1e-5))
 def test_loss_has_finite_gradients_and_same_team_negatives(self):
  model=MetricHead();z=model(torch.randn(8,384));labels=torch.tensor([0,0,1,1,2,2,3,3]);teams=torch.tensor([0,0,0,0,1,1,1,1])
  loss,terms=objective(z,labels,teams);loss.backward();self.assertTrue(torch.isfinite(loss));self.assertEqual(terms['same_team_anchors'],8);self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
 def test_opponents_not_counted_as_same_team_negatives(self):
  z=torch.nn.functional.normalize(torch.randn(4,128),dim=1);_,terms=objective(z,torch.tensor([0,0,1,1]),torch.tensor([0,0,1,1]));self.assertEqual(terms['same_team_anchors'],0)
 def test_multiple_views_do_not_create_duplicate_id_runner_up(self):
  gallery=[dict(sequence='x',identity='a',team='T'),dict(sequence='x',identity='a',team='T'),dict(sequence='x',identity='b',team='T')]
  q=[dict(sequence='x',frame=1,identity='a',team='T')]
  r=rank(gallery,np.array([[1.,0.],[.99,.01],[0.,1.]]),q,np.array([[1.,0.]]),4)[0]
  self.assertTrue(r['correct']);self.assertEqual(r['margin'],1.)
 def test_unknown_acceptance_is_an_error(self):
  r=[dict(known=False,correct=False,same_team_error=False,margin=.8)]
  s=summarize(r,.1);self.assertEqual(s['accepted_unknown'],1);self.assertEqual(s['accepted_precision'],0.)
 def test_identity_splits_and_gallery_time(self):
  rows=json.loads((R/'features.json').read_text());train={r['identity'] for r in rows if r['split']=='train'};val={r['identity'] for r in rows if r['split']=='validation'}
  g=json.loads((R/'gallery.json').read_text());test={r['identity'] for r in g}
  self.assertFalse(train&val);self.assertFalse((train|val)&test);self.assertTrue(all(r['frame']<=120 for r in g));self.assertTrue(all(r['sequence']!='CHI_VS_TOR_004' for r in rows))
if __name__=='__main__':unittest.main()
