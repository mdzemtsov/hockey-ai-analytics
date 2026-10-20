import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import evaluate


def fixture(n=4):
    gt = [dict(frame_index=i, pts_s=i/30, target_id='RED_TEST_01',
               status='visible', x=0, y=0, w=10, h=20,
               provenance='manual', reviewer='SYNTHETIC_FIXTURE') for i in range(n)]
    pred = [dict(frame_index=i, pts_s=i/30, state='TRACKING',
                 x=0, y=0, w=10, h=20, observed_id='RED_TEST_01') for i in range(n)]
    return gt, pred


class MetricsTest(unittest.TestCase):
    def test_perfect(self):
        s, _ = evaluate(*fixture())
        self.assertEqual(s['coverage'], 1)
        self.assertEqual(s['mean_iou_conditional'], 1)
        self.assertEqual(s['mean_center_error_px_conditional'], 0)

    def test_loss_penalizes_coverage(self):
        g, p = fixture()
        for i in (1, 2):
            p[i].update(state='LOST', x='', y='', w='', h='', observed_id='')
        s, _ = evaluate(g, p)
        self.assertEqual(s['coverage'], .5)
        self.assertEqual(s['longest_missing_visible_run_frames'], 2)
        self.assertEqual(s['iou_ge_05_fraction_all_eligible'], .5)

    def test_switch_and_return(self):
        g, p = fixture()
        p[1]['observed_id'] = p[2]['observed_id'] = 'OTHER_01'
        s, _ = evaluate(g, p)
        self.assertEqual(s['wrong_identity_frames_reviewed'], 2)
        self.assertEqual(s['identity_transitions_contiguous_reviewed'], 2)

    def test_unreviewed_is_not_zero_switches(self):
        g, p = fixture()
        for r in p:
            r['observed_id'] = ''
        s, _ = evaluate(g, p)
        self.assertIsNone(s['identity_transitions_contiguous_reviewed'])
        self.assertFalse(s['identity_audit_complete'])

    def test_exclude_interpolation_and_unknown(self):
        g, p = fixture()
        g[0]['provenance'] = 'interpolated'
        g[1]['status'] = 'identity_unknown'
        s, _ = evaluate(g, p)
        self.assertEqual(s['eligible_visible_manual_frames'], 2)
        self.assertEqual(s['predictions_on_unscored_frames'], 2)

    def test_center_error(self):
        g, p = fixture(1)
        p[0].update(x=3, y=4)
        s, _ = evaluate(g, p)
        self.assertEqual(s['mean_center_error_px_conditional'], 5)
        self.assertEqual(s['mean_center_error_height_conditional'], .25)

    def test_reject_bad_sync_missing_duplicate_and_empty(self):
        g, p = fixture()
        p[0]['pts_s'] = .01
        with self.assertRaises(ValueError): evaluate(g, p)
        g, p = fixture()
        with self.assertRaises(ValueError): evaluate(g, p[:-1])
        with self.assertRaises(ValueError): evaluate(g, p+[p[0]])
        with self.assertRaises(ValueError): evaluate([], [])

    def test_reject_coordinates_during_abstention(self):
        g, p = fixture()
        p[0]['state'] = 'LOST'
        with self.assertRaises(ValueError): evaluate(g, p)


if __name__ == '__main__':
    unittest.main()
