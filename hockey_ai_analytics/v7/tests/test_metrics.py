import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark import ious,score
class MetricsTests(unittest.TestCase):
 def row(self,f,tid):return dict(frame=f,id=tid,x1=0,y1=0,x2=10,y2=10)
 def gt(self):return np.array([[1,1,0,0,10,10,1,1,1],[2,1,0,0,10,10,1,1,1]],float)
 def test_perfect(self):
  m=score([self.row(1,7),self.row(2,7)],self.gt());self.assertEqual(m['idf1'],1);self.assertEqual(m['num_switches'],0)
 def test_identity_switch(self):
  m=score([self.row(1,7),self.row(2,8)],self.gt());self.assertEqual(m['num_switches'],1);self.assertEqual(m['recall'],1);self.assertLess(m['idf1'],1)
 def test_loss(self):
  m=score([self.row(1,7)],self.gt());self.assertEqual(m['num_misses'],1);self.assertEqual(m['recall'],.5)
 def test_false_positive(self):
  r=self.row(2,9);r.update(x1=100,y1=100,x2=110,y2=110);m=score([self.row(1,7),r],self.gt());self.assertEqual(m['num_false_positives'],1)
 def test_empty_boxes(self):self.assertEqual(ious([],[]).shape,(0,0))
if __name__=='__main__':unittest.main()
