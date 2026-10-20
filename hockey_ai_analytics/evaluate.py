"""Single-target evaluator. No video tracking or identity inference is performed."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


def read_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def index_rows(rows):
    indexed = {}
    for row in rows:
        i = int(row['frame_index'])
        if i < 0 or i in indexed:
            raise ValueError('Negative or duplicate frame_index')
        if not math.isfinite(float(row['pts_s'])):
            raise ValueError('Nonfinite PTS')
        indexed[i] = row
    return indexed


def box(row):
    values = [row.get(k, '') for k in ('x', 'y', 'w', 'h')]
    if all(v == '' for v in values):
        return None
    if any(v == '' for v in values):
        raise ValueError('Partial bbox')
    b = tuple(map(float, values))
    if not all(math.isfinite(v) for v in b) or min(b[2:]) <= 0:
        raise ValueError('Invalid bbox')
    return b


def ratio(a, b):
    return a / b if b else None


def longest_run(indices):
    best = run = 0
    prev = None
    for i in indices:
        run = run + 1 if prev is not None and i == prev + 1 else 1
        best = max(best, run)
        prev = i
    return best


def evaluate(gt_rows, pred_rows):
    gt, pred = index_rows(gt_rows), index_rows(pred_rows)
    if not gt or set(gt) != set(pred):
        raise ValueError('Require nonempty, identical GT/pred frame sets')
    frames = sorted(gt)
    if frames != list(range(frames[-1] + 1)):
        raise ValueError('Require every frame from zero')
    valid_states = {'TRACKING', 'LOST', 'AMBIGUOUS', 'NOT_STARTED'}
    visible = matched = reviewed = wrong = switches = gaps = 0
    abstentions = missing = interpolated = unknown_emissions = 0
    errors, normalized, ious, detail, covered, misses, false_ids = [], [], [], [], [], [], []
    previous_id = None
    previous_pts = -math.inf
    for i in frames:
        g, p = gt[i], pred[i]
        pts = float(g['pts_s'])
        if pts <= previous_pts or abs(pts - float(p['pts_s'])) > .001:
            raise ValueError('PTS order or synchronization mismatch')
        previous_pts = pts
        if g['status'] not in {'visible', 'occluded', 'out_of_view', 'identity_unknown'}:
            raise ValueError('Unknown GT status')
        if g['provenance'] not in {'manual', 'interpolated'}:
            raise ValueError('Unknown provenance')
        if not g['target_id'] or not g.get('reviewer'):
            raise ValueError('Require target_id and annotation reviewer')
        if p['state'] not in valid_states:
            raise ValueError('Unknown prediction state')
        gb, pb = box(g), box(p)
        emitted = p['state'] == 'TRACKING'
        if emitted != (pb is not None):
            raise ValueError('TRACKING needs bbox; abstention must have none')
        if g['status'] == 'visible' and gb is None:
            raise ValueError('Visible GT requires bbox')
        if p.get('observed_id') and not emitted:
            raise ValueError('observed_id requires prediction')
        eligible = g['status'] == 'visible' and g['provenance'] == 'manual'
        interpolated += g['provenance'] == 'interpolated'
        abstentions += not emitted
        row = {'frame_index': i, 'pts_s': pts, 'eligible': eligible,
               'state': p['state'], 'center_error_px': None,
               'center_error_height': None, 'iou': None, 'identity_correct': None}
        if eligible:
            visible += 1
            if emitted:
                matched += 1
                covered.append(i)
                x, y, w, h = gb
                a, b, c, d = pb
                error = math.hypot(x+w/2-a-c/2, y+h/2-b-d/2)
                intersection = max(0, min(x+w, a+c)-max(x, a)) * max(0, min(y+h, b+d)-max(y, b))
                iou = intersection / (w*h+c*d-intersection)
                errors.append(error)
                normalized.append(error/h)
                ious.append(iou)
                row.update(center_error_px=error, center_error_height=error/h, iou=iou)
                identity = p.get('observed_id', '')
                if identity:
                    reviewed += 1
                    correct = identity == g['target_id']
                    row['identity_correct'] = correct
                    wrong += not correct
                    if not correct:
                        false_ids.append(i)
                    if previous_id is not None and identity != previous_id:
                        switches += 1
                    previous_id = identity
                else:
                    gaps += 1
                    previous_id = None
            else:
                missing += 1
                misses.append(i)
                previous_id = None
        else:
            unknown_emissions += emitted
            previous_id = None
        detail.append(row)
    summary = {
        'frames': len(frames), 'eligible_visible_manual_frames': visible,
        'interpolated_frames_excluded': interpolated,
        'predicted_on_eligible_frames': matched,
        'coverage': ratio(matched, visible),
        'missing_visible_frames': missing,
        'longest_missing_visible_run_frames': longest_run(misses),
        'longest_prediction_run_frames': longest_run(covered),
        'abstained_frames_all': abstentions,
        'predictions_on_unscored_frames': unknown_emissions,
        'mean_center_error_px_conditional': mean(errors) if errors else None,
        'mean_center_error_height_conditional': mean(normalized) if normalized else None,
        'mean_iou_conditional': mean(ious) if ious else None,
        'iou_ge_05_fraction_all_eligible': ratio(sum(v >= .5 for v in ious), visible),
        'identity_reviewed_predicted_frames': reviewed,
        'identity_audit_complete': matched > 0 and reviewed == matched,
        'identity_audit_gap_frames': gaps,
        'wrong_identity_frames_reviewed': wrong if reviewed else None,
        'wrong_identity_frame_indices': false_ids,
        'identity_correct_fraction_reviewed': ratio(reviewed-wrong, reviewed),
        'identity_transitions_contiguous_reviewed': switches if reviewed else None,
        'release_verdict': 'NOT_CERTIFIED_BY_THIS_EVALUATOR',
    }
    return summary, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt', required=True)
    parser.add_argument('--pred', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    summary, detail = evaluate(read_rows(args.gt), read_rows(args.pred))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    with (out/'per_frame.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)


if __name__ == '__main__':
    main()
