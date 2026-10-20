import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from coach import jersey_color,assess,validate_request

def obj(x,y,color='RED',kind='player',confidence=.9):return dict(kind=kind,bbox=[x,y,20,40],color=color,confidence=confidence,color_fraction=.8)
class CoachTests(unittest.TestCase):
    def test_colors(self):
        for bgr,color in [((0,0,230),'RED'),((230,20,20),'BLUE'),((230,230,230),'WHITE'),((35,35,35),'DARK'),((0,200,230),'YELLOW'),((0,180,0),'GREEN')]:
            im=np.full((100,100,3),bgr,dtype=np.uint8)
            self.assertEqual(jersey_color(im,[0,0,100,100])[0],color)
    def test_empty_crop_unknown(self):self.assertEqual(jersey_color(np.zeros((10,10,3),np.uint8),[20,20,5,5])[0],'UNKNOWN')
    def test_puck_missing_prevents_judgment(self):
        r=assess([obj(10,10),obj(100,10),obj(10,200,'BLUE'),obj(100,200,'BLUE')],'AUTO')
        self.assertEqual(r['puck_status'],'UNKNOWN');self.assertEqual(r['arrows'],[]);self.assertEqual(r['possession'],'UNKNOWN')
    def test_multiple_pucks_abstain(self):
        r=assess([obj(10,10,kind='puck'),obj(100,100,kind='puck')],'AUTO');self.assertEqual(r['puck_status'],'UNKNOWN');self.assertEqual(r['arrows'],[])
    def test_two_groups_only_not_team_truth(self):
        r=assess([obj(0,0),obj(100,0),obj(0,200,'BLUE'),obj(100,200,'BLUE')],'AUTO');self.assertEqual(r['teams_status'],'COLOR_HYPOTHESIS')
    def test_conditional_lane_not_good_play(self):
        objects=[obj(10,10),obj(150,10),obj(10,200,'BLUE'),obj(150,200,'BLUE'),obj(10,12,kind='puck')]
        r=assess(objects,'RED');self.assertEqual(len(r['arrows']),1);self.assertEqual(r['evaluation'],'CONDITIONAL_SUPPORT_OPTION');self.assertEqual(r['possession'],'UNKNOWN')
        objects[2]['bbox']=[80,10,20,40];self.assertEqual(assess(objects,'RED')['arrows'],[])
    def test_selected_player_does_not_inherit_other_players_lane(self):
        objects=[obj(10,10),obj(150,10),obj(10,200,'BLUE'),obj(150,200,'BLUE'),obj(10,12,kind='puck')]
        self.assertEqual(assess(objects,'RED',selected=2)['arrows'],[])
    def test_unknown_people_block_lane(self):
        scene=[obj(10,10),obj(150,10),obj(10,200,'BLUE'),obj(150,200,'BLUE'),obj(10,12,kind='puck')]
        self.assertTrue(assess(scene,'RED')['arrows'])
        for kind in ['player','goalie','referee']:
            with self.subTest(kind=kind):
                self.assertEqual(assess(scene+[obj(80,10,'UNKNOWN',kind)],'RED')['arrows'],[])
    def test_requested_team_never_silently_replaced(self):
        scene=[obj(10,10),obj(150,10),obj(10,200,'BLUE'),obj(150,200,'BLUE'),obj(10,12,kind='puck')]
        self.assertEqual(assess(scene,'WHITE')['arrows'],[])
    def test_third_color_does_not_disappear(self):
        scene=[obj(10,10),obj(150,10),obj(10,200,'BLUE'),obj(150,200,'BLUE'),obj(10,12,kind='puck'),obj(400,400,'WHITE')]
        self.assertEqual(assess(scene,'AUTO')['teams_status'],'UNKNOWN')
    def test_validation(self):
        meta=dict(duration=60,frame_count=60,pts=list(range(60)),width=200,height=100)
        for body in [dict(start=0,end=40),dict(start=float('nan'),end=10),dict(start=5,end=2),dict(start=0,end=10,mode='player'),dict(start=0,end=10,team='madeup')]:
            with self.assertRaises((ValueError,TypeError)):validate_request(body,meta)
        self.assertEqual(validate_request(dict(start=1,end=5),meta)['mode'],'team')
if __name__=='__main__':unittest.main()
