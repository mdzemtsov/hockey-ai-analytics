import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation import evaluate


class EvaluationTests(unittest.TestCase):
    def test_counts_after_frame_300(self):
        gt=np.array([[1,1,0,0,10,10],[301,1,0,0,10,10]],float)
        pred=[dict(frame=1,id=1,x1=0,y1=0,x2=10,y2=10)]
        result=evaluate(pred,gt,301)
        self.assertEqual(result["num_frames"],301)
        self.assertEqual(result["num_misses"],1)
        self.assertEqual(result["recall"],.5)

    def test_switch_and_absent_false_positive(self):
        gt=np.array([[1,1,0,0,10,10],[2,1,0,0,10,10]],float)
        pred=[dict(frame=f,id=i,x1=0,y1=0,x2=10,y2=10) for f,i in [(1,1),(2,2),(3,2)]]
        result=evaluate(pred,gt,3)
        self.assertEqual(result["num_switches"],1)
        self.assertEqual(result["num_false_positives"],1)

    def test_duplicate_ids_rejected(self):
        p=dict(frame=1,id=1,x1=0,y1=0,x2=10,y2=10)
        with self.assertRaises(ValueError):
            evaluate([p,p],np.empty((0,6)),1)

    def test_outside_interval_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([dict(frame=2,id=1)],np.empty((0,6)),1)


if __name__=="__main__":
    unittest.main()
