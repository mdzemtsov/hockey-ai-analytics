import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parent/"v7")]
from stable_fusion import StableFusion
from benchmark import ious


def box(tid,x):
    return dict(id=tid,x1=x,y1=0,x2=x+10,y2=20,confidence=.9,cls=4)


class FusionTests(unittest.TestCase):
    def test_three_confirmations_then_persist_on_missing_mask(self):
        f=StableFusion([1],ious)
        for frame in [1,2]:
            self.assertEqual(f.update(frame,[box(7,0)],[box(1000,0)])[0]["id"],7)
        self.assertEqual(f.update(3,[box(7,0)],[box(1000,0)])[0]["id"],1)
        self.assertEqual(f.update(4,[box(7,0)],[])[0]["id"],1)

    def test_missing_observation_resets_confirmation(self):
        f=StableFusion([1],ious)
        f.update(1,[box(7,0)],[box(1000,0)])
        f.update(2,[box(7,0)],[])
        self.assertEqual(f.update(3,[box(7,0)],[box(1000,0)])[0]["id"],7)

    def test_swapped_pair_changes_together_without_duplicate(self):
        f=StableFusion([1,2],ious)
        for frame in [1,2,3]:
            out=f.update(frame,[box(1,30),box(2,0)],[box(1000,0),box(1001,30)])
        self.assertEqual({r["input_id"]:r["id"] for r in out},{1:2,2:1})

    def test_absent_prompt_does_not_shift_identity_index(self):
        f=StableFusion([10,20],ious)
        for frame in [1,2,3]:
            out=f.update(frame,[box(7,0)],[box(1001,0)])
        self.assertEqual(out[0]["id"],20)

    def test_duplicate_claim_marked_unknown(self):
        f=StableFusion([1],ious)
        f.alias[7]=1
        out=f.update(1,[box(1,0),box(7,30)],[])
        self.assertEqual(len({r["id"] for r in out}),2)
        self.assertEqual(sum(r["association_state"]=="CONFLICT_UNKNOWN" for r in out),1)


if __name__=="__main__":unittest.main()
