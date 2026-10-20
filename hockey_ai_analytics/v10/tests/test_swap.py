import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parent/"v7")]
from reciprocal_swap import ReciprocalSwap
from benchmark import ious


def box(tid,x):
    return dict(id=tid,x1=x,y1=0,x2=x+40,y2=100,confidence=.9,cls=4)


class SwapTests(unittest.TestCase):
    def setUp(self):
        self.first=np.zeros((100,100,3),np.uint8)
        self.first[:,:40]=(200,30,20);self.first[:,60:]=240
        self.seeds=[box(1,0),box(2,60)]

    def test_reciprocal_color_swap_only_after_three(self):
        m=ReciprocalSwap(self.seeds,self.first,ious)
        # Physical colors unchanged; detector IDs reversed.
        for frame in [1,2,3]:
            out=m.update(frame,self.first,[box(2,0),box(1,60)],[box(1000,0),box(1001,60)])
            ids={r["input_id"]:r["id"] for r in out}
            self.assertEqual(ids,{2:1,1:2} if frame==3 else {2:2,1:1})
        out=m.update(4,self.first,[box(2,0)],[])
        self.assertEqual(out[0]["id"],1)

    def test_no_unilateral_reassignment(self):
        m=ReciprocalSwap(self.seeds,self.first,ious)
        for frame in range(1,8):
            out=m.update(frame,self.first,[box(2,0)],[box(1000,0)])
            self.assertEqual(out[0]["id"],2)

    def test_wrong_color_sam_pair_does_not_override_baseline(self):
        m=ReciprocalSwap(self.seeds,self.first,ious)
        for frame in range(1,8):
            out=m.update(frame,self.first,self.seeds,[box(1000,60),box(1001,0)])
            self.assertEqual({r["input_id"]:r["id"] for r in out},{1:1,2:2})


if __name__=="__main__":unittest.main()
