import hashlib,io,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.fetch_assets import fetch
class AssetFetchTests(unittest.TestCase):
    def asset(self):return dict(path='weights/model.pt',download_url='https://example.invalid/model',sha256=hashlib.sha256(b'test').hexdigest())
    def test_verified_download_and_reuse(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(fetch(self.asset(),root,lambda *a,**k:io.BytesIO(b'test')),'downloaded and verified')
            self.assertEqual(fetch(self.asset(),root,lambda *a,**k:self.fail('Unexpected network')),'already verified')
    def test_bad_download_not_installed(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):fetch(self.asset(),root,lambda *a,**k:io.BytesIO(b'bad'))
            self.assertFalse((Path(root)/self.asset()['path']).exists());self.assertFalse(list(Path(root).rglob('*.part')))
    def test_existing_mismatch_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/self.asset()['path'];p.parent.mkdir();p.write_bytes(b'keep')
            with self.assertRaises(ValueError):fetch(self.asset(),root)
            self.assertEqual(p.read_bytes(),b'keep')
    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):fetch({**self.asset(),'path':'../escape.pt'},root)
if __name__=='__main__':unittest.main()
