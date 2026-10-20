import sys,unittest,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gate import ManualGate,bind,validate_selections

def obj(t=1,x=10,team='RED'):
    return dict(track_id=t,x=x,y=10,w=20,h=40,detector_conf=.9,team_hypothesis=team)
def anchor(i=0,confirmed=False):return dict(frame_index=i,bbox=[10,10,20,40],confirmed_same_player=confirmed)
class GateTests(unittest.TestCase):
    def test_white_player(self):
        g=ManualGate([anchor()]);self.assertEqual(g.update(0,0,[obj(team='NONRED_UNKNOWN_ROLE')])['state'],'TRACKING')
    def test_no_nearest_neighbour_rebind(self):
        g=ManualGate([anchor()]);g.update(0,0,[obj()]);r=g.update(1,.03,[obj(t=2)])
        self.assertEqual(r['state'],'UNKNOWN');self.assertEqual(r['x'],'')
    def test_ambiguous_initial_selection(self):
        self.assertEqual(bind(anchor(),[obj(),obj(t=2,x=11)])[1],'SELECTION_CONTAINS_MULTIPLE_PLAYERS')
    def test_empty_selection(self):self.assertIsNone(bind(anchor(),[])[0])
    def test_overlap_hides_marker(self):
        g=ManualGate([anchor()]);g.update(0,0,[obj()]);r=g.update(1,.03,[obj(),obj(2,15)])
        self.assertEqual(r['state'],'UNKNOWN');self.assertEqual(r['x'],'')
    def test_same_id_cannot_recover_without_confirmation(self):
        g=ManualGate([anchor()]);g.update(0,0,[obj()]);g.update(1,.03,[])
        self.assertEqual([g.update(i,i/30,[obj()])['state'] for i in [2,3,4]],['UNKNOWN','UNKNOWN','UNKNOWN'])
    def test_explicit_manual_rebind_logged(self):
        g=ManualGate([anchor(),anchor(2,True)]);g.update(0,0,[obj()]);g.update(1,.03,[])
        r=g.update(2,.06,[obj(7)]);self.assertEqual(r['model_track_id'],7);self.assertTrue(r['manual_anchor']);self.assertEqual(len(g.events),2)
    def test_overlap_latches_even_after_separation(self):
        g=ManualGate([anchor()]);g.update(0,0,[obj()])
        g.update(1,.03,[obj(),obj(2,15)])
        for i in range(2,20):
            row=g.update(i,i/30,[obj()])
            self.assertEqual(row['identity_status'],'UNKNOWN')
            self.assertEqual(row['x'],'')
    def test_confirmation_does_not_override_ambiguity(self):
        g=ManualGate([anchor(),anchor(2,True)])
        g.update(0,0,[obj()]);g.update(1,.03,[])
        self.assertNotEqual(g.update(2,.06,[obj(),obj(2,15)])['state'],'TRACKING')
    def test_confirmation_requires_boolean(self):
        a=anchor(2);a['confirmed_same_player']='false'
        with self.assertRaises(ValueError):validate_selections([anchor(),a],dict(frame_count=10,width=100,height=100))
    def test_validation(self):
        m=dict(frame_count=10,width=100,height=100)
        for a in [[anchor(),anchor(2)], [anchor(10)], [dict(frame_index=0,bbox=[0,0,float('nan'),20])], [dict(frame_index=0,bbox=[90,0,20,20])], [anchor(2),anchor(1,True)]]:
            with self.subTest(a=a):
                with self.assertRaises(ValueError):validate_selections(a,m)
    def test_preselection_has_no_marker(self):
        self.assertEqual(ManualGate([anchor(5)]).update(0,0,[obj()])['state'],'NOT_STARTED')
if __name__=='__main__':unittest.main()
