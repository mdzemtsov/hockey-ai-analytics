import sys,unittest
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));from motion_check import check
class MotionTests(unittest.TestCase):
 def images(self):
  a=np.zeros((160,200),np.uint8);a[40:100,50:100]=np.random.default_rng(3).integers(0,255,(60,50),dtype=np.uint8);b=cv2.warpAffine(a,np.float32([[1,0,8],[0,1,3]]),(200,160));return a,b
 def test_consistent_translation(self):
  a,b=self.images();self.assertEqual(check(a,b,[50,40,100,100],[58,43,108,103])['status'],'AGREES')
 def test_jump_rejected(self):
  a,b=self.images();self.assertEqual(check(a,b,[50,40,100,100],[110,43,160,103])['status'],'DISAGREES')
 def test_no_features_unknown(self):
  a=np.zeros((160,200),np.uint8);self.assertEqual(check(a,a,[50,40,100,100],[50,40,100,100])['status'],'UNKNOWN')
 def test_absence_unknown(self):
  a,b=self.images();self.assertEqual(check(a,b,None,[58,43,108,103])['status'],'UNKNOWN')
if __name__=='__main__':unittest.main()
