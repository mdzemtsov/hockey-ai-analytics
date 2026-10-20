from pathlib import Path
import sys,unittest,collections
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from score_recovery import evaluate


class Tests(unittest.TestCase):
    def test_abstention_is_not_success(self):
        _,m=evaluate([None]*5,[dict(visible=True,box=[0,0,10,10])]*5,collections.defaultdict(list))
        self.assertEqual(m["correct_visible_coverage"],0)
        self.assertTrue(m["recovery_censored"])

    def test_wrong_person_and_three_frame_recovery(self):
        b=[0,0,10,10];other=[20,0,30,10]
        rows,m=evaluate([b,other,None,b,b,b],[dict(visible=True,box=b)]*6,collections.defaultdict(list,{2:[other]}))
        self.assertEqual(m["wrong_person_frames"],1)
        self.assertEqual(m["first_stable_recovery_frame"],4)
        self.assertAlmostEqual(m["recovery_delay_s"],2/30)

    def test_absence_does_not_count_as_recovered(self):
        _,m=evaluate([None]*4,[dict(visible=False,box=None)]*4,collections.defaultdict(list))
        self.assertIsNone(m["correct_visible_coverage"])
        self.assertIsNone(m["first_stable_recovery_frame"])

    def test_partial_recovery_not_stable(self):
        b=[0,0,10,10]
        _,m=evaluate([None,b,b],[dict(visible=True,box=b)]*3,collections.defaultdict(list))
        self.assertTrue(m["recovery_censored"])

    def test_frame_dropping_rejected(self):
        with self.assertRaises(ValueError):evaluate([], [dict(visible=False,box=None)],{})


if __name__=="__main__":unittest.main()
