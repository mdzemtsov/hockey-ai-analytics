import unittest
from target_gate import TargetGate


def person(tid=4,x=180,y=280,w=30,h=50,team='RED'):
    return dict(track_id=tid,x=x,y=y,w=w,h=h,team_hypothesis=team,detector_conf=.9)


class GateTests(unittest.TestCase):
    def test_seed_and_next_frame_no_artificial_gap(self):
        g=TargetGate();self.assertEqual(g.update(90,3.,[person()])['state'],'TRACKING')
        self.assertEqual(g.update(91,3.03,[person()])['state'],'TRACKING')

    def test_neighbor_cannot_replace_missing_bound_id(self):
        g=TargetGate();g.update(90,3.,[person()])
        self.assertEqual(g.update(91,3.03,[person(tid=99)])['state'],'LOST')
        self.assertEqual(g.bound_id,4)

    def test_red_neighbor_abstention_has_no_box(self):
        g=TargetGate();g.update(90,3.,[person()])
        r=g.update(91,3.03,[person(),person(tid=99,x=205)])
        self.assertEqual(r['state'],'AMBIGUOUS');self.assertEqual(r['x'],'')

    def test_white_overlap_is_also_uncertain(self):
        g=TargetGate();g.update(90,3.,[person()])
        self.assertEqual(g.update(91,3.03,[person(),person(tid=99,x=190,team='NONRED_UNKNOWN_ROLE')])['reason'],'OVERLAPPING_PERSON_IDENTITY_UNCERTAIN')

    def test_recovery_requires_three_frames_same_id(self):
        g=TargetGate();g.update(90,3.,[person()]);g.update(91,3.03,[])
        self.assertEqual([g.update(i,i/30,[person()])['state'] for i in [92,93,94]],['AMBIGUOUS','AMBIGUOUS','TRACKING'])

    def test_ambiguous_seed_is_never_auto_bound_later(self):
        g=TargetGate();g.update(90,3.,[person(),person(tid=99)])
        self.assertEqual(g.update(91,3.03,[person()])['state'],'LOST')
        self.assertIsNone(g.bound_id)


if __name__=='__main__':unittest.main()
