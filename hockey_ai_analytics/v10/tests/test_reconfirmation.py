import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reconfirmation_gate import ReconfirmationGate


def obj(tid=1,x=0):
    return dict(track_id=tid,x=x,y=0,w=20,h=40,detector_conf=.9,team_hypothesis="UNKNOWN")


class ReconfirmationTests(unittest.TestCase):
    def test_missing_identity_cannot_resume_without_anchor(self):
        g=ReconfirmationGate([dict(frame_index=0,bbox=[0,0,20,40])])
        self.assertEqual(g.update(0,0,[obj()])["state"],"TRACKING")
        self.assertEqual(g.update(1,.03,[])["state"],"NEEDS_SELECTION")
        for f in range(2,9):
            row=g.update(f,f/30,[obj()])
            self.assertEqual(row["state"],"NEEDS_SELECTION")
            self.assertEqual(row["x"],"")

    def test_explicit_anchor_can_restore(self):
        g=ReconfirmationGate([dict(frame_index=0,bbox=[0,0,20,40]),
                             dict(frame_index=3,bbox=[50,0,20,40],confirmed_same_player=True)])
        g.update(0,0,[obj()]);g.update(1,.03,[])
        self.assertEqual(g.update(3,.1,[obj(2,50)])["state"],"TRACKING")

    def test_overlap_latches(self):
        g=ReconfirmationGate([dict(frame_index=0,bbox=[0,0,20,40])])
        g.update(0,0,[obj()])
        self.assertEqual(g.update(1,.03,[obj(),obj(2,5)])["state"],"NEEDS_SELECTION")
        self.assertEqual(g.update(2,.06,[obj()])["state"],"NEEDS_SELECTION")


if __name__=="__main__":unittest.main()
