import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from team_calibration import TeamCalibration


class TeamTests(unittest.TestCase):
    def setUp(self):
        self.blue=np.full((100,50,3),(200,30,20),np.uint8)
        self.white=np.full((100,50,3),240,np.uint8)
        self.box=[0,0,50,100]

    def test_one_team_never_creates_second_cluster(self):
        c=TeamCalibration();c.add("TEAM_A",self.blue,self.box)
        self.assertEqual(c.classify(self.blue,self.box)["team"],"UNKNOWN")

    def test_duplicate_examples_rejected_as_calibration(self):
        c=TeamCalibration()
        for label in ["TEAM_A","TEAM_B"]:c.add(label,self.blue,self.box)
        self.assertEqual(c.readiness(),"EXAMPLES_TOO_SIMILAR")
        self.assertEqual(c.classify(self.white,self.box)["team"],"UNKNOWN")

    def test_distinct_examples_and_referee(self):
        c=TeamCalibration();c.add("TEAM_A",self.blue,self.box);c.add("TEAM_B",self.white,self.box)
        self.assertEqual(c.classify(self.blue,self.box)["team"],"TEAM_A")
        self.assertEqual(c.classify(self.white,self.box)["team"],"TEAM_B")
        self.assertEqual(c.classify(self.blue,self.box,6)["team"],"NOT_APPLICABLE")

    def test_empty_exemplar_not_accepted(self):
        with self.assertRaises(ValueError):
            TeamCalibration().add("TEAM_A",self.blue,[100,100,120,120])


if __name__=="__main__":unittest.main()
