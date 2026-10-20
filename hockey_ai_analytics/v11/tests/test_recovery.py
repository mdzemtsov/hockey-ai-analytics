import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recovery import Recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / "recovery_config.json").read_text())
        self.feature = np.array([1., 0.])
        self.box = [100., 100., 140., 200.]
        self.engine = Recovery(self.cfg, self.box, self.feature, 30)
        self.engine.update([self.candidate(1)])

    def candidate(self, tid, feature=None, box=None):
        return dict(id=tid, cls=4, box=box or self.box,
                    feature=self.feature if feature is None else feature)

    def test_fragmented_id_recovers_only_after_three_frames(self):
        self.assertIsNone(self.engine.update([])["box"])
        self.assertEqual(self.engine.update([self.candidate(9)])["state"], "CONFIRMING")
        self.assertIsNone(self.engine.update([self.candidate(9)])["box"])
        self.assertEqual(self.engine.update([self.candidate(9)])["state"], "RECOVERED")

    def test_reused_id_does_not_bypass_appearance(self):
        result = self.engine.update([self.candidate(1, np.array([0., 1.]))])
        self.assertIsNone(result["box"])

    def test_two_indistinguishable_candidates_abstain(self):
        result = self.engine.update([self.candidate(2), self.candidate(3)])
        self.assertEqual(result["state"], "AMBIGUOUS")
        self.assertIsNone(result["box"])

    def test_confirmation_must_be_consecutive(self):
        self.engine.update([self.candidate(9)])
        self.engine.update([])
        self.engine.update([self.candidate(9)])
        self.assertIsNone(self.engine.update([self.candidate(9)])["box"])

    def test_far_same_uniform_is_rejected(self):
        self.assertIsNone(self.engine.update([self.candidate(2, box=[700, 100, 740, 200])])["box"])

    def test_expired_target_requires_explicit_selection(self):
        for _ in range(46):
            self.engine.update([])
        self.assertEqual(self.engine.update([self.candidate(1)])["state"], "RESELECT_REQUIRED")

    def test_failed_selection_cannot_silently_reinitialize(self):
        engine = Recovery(self.cfg, self.box, self.feature, 30)
        self.assertEqual(engine.update([])["state"], "SELECTION_FAILED")
        self.assertEqual(engine.update([self.candidate(1)])["state"], "RESELECT_REQUIRED")

    def test_class_mismatch_is_rejected(self):
        row = self.candidate(1)
        row["cls"] = 6
        self.assertIsNone(self.engine.update([row])["box"])


if __name__ == "__main__":
    unittest.main()
