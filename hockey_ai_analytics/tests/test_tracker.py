import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tracker_v4 import select,run


class SafetyGateTest(unittest.TestCase):
    def test_close_candidates_abstain(self):
        candidate,reason=select([{'torso':np.array([1.,0.])},{'torso':np.array([3.,0.])}],np.array([0.,0.]))
        self.assertIsNone(candidate)
        self.assertEqual(reason,'AMBIGUOUS_RED_CANDIDATES')

    def test_distant_candidate_is_not_reassigned(self):
        candidate,_=select([{'torso':np.array([100.,0.])}],np.array([0.,0.]))
        self.assertIsNone(candidate)

    def test_loss_latches_even_if_red_returns(self):
        frames=[np.full((464,848,3),200,np.uint8) for _ in range(93)]
        frames[92][284:305,187:204]=[0,0,160]
        rows=run(frames,[i/30 for i in range(93)])
        self.assertEqual(rows[90]['state'],'LOST')
        self.assertEqual(rows[92]['state'],'LOST')
        self.assertEqual(rows[92]['x'],'')


if __name__=='__main__':unittest.main()
