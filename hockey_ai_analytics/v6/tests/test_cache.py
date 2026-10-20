import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import read_cache,save_cache
class CacheTests(unittest.TestCase):
    def test_invalidations(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'detections.json';sig={'frame_count':2,'source':'A','model':'A'}
            frames=[[],[]];save_cache(p,sig,frames)
            self.assertEqual(read_cache(p,sig),frames)
            for key in ['source','model']:
                self.assertIsNone(read_cache(p,{**sig,key:'B'}))
            data=json.loads(p.read_text());data['frames'][0]=[{'track_id':99}]
            p.write_text(json.dumps(data));self.assertIsNone(read_cache(p,sig))
            p.write_text(json.dumps(frames));self.assertIsNone(read_cache(p,sig))
            p.write_text('{');self.assertIsNone(read_cache(p,sig))
if __name__=='__main__':unittest.main()
