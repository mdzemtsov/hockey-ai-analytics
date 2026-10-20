"""Build the review index and evidence hashes from saved benchmark results."""
from pathlib import Path
import collections
import hashlib
import html
import json

ROOT = Path(__file__).resolve().parent


def main():
    rows = json.loads((ROOT / "baseline_summary.json").read_text())
    if (ROOT / "recovery_summary.json").exists():
        rows += [dict(method="recovery_v1", **r) for r in json.loads((ROOT / "recovery_summary.json").read_text())]
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["case"]].append(row)
    sections = []
    for case, results in grouped.items():
        table = "".join(
            f"<tr><td>{html.escape(r['method'])}</td>"
            f"<td>{r['correct']}/{r['visible_frames']}</td>"
            f"<td>{r['wrong_person_frames']}</td>"
            f"<td>{'yes' if r['seed_bound'] else 'FAILED'}</td></tr>"
            for r in results
        )
        sections.append(
            f"<section><h2>{html.escape(case)}</h2>"
            f'<video controls preload="metadata" src="cases/{case}/source.mp4"></video>'
            f"<p>Unannotated source video. Reference boxes below are publisher labels.</p>"
            f'<img loading="lazy" src="cases/{case}/contact_sheet.jpg">'
            f"<table><tr><th>Method</th><th>Correct / visible frames</th>"
            f"<th>Wrong-person frames</th><th>Seed bound</th></tr>{table}</table>"
            f'<p><a href="cases/{case}/reference.csv">Reference CSV</a></p></section>'
        )
        video = f"results/recovery/{case}/comparison.mp4"
        if (ROOT / video).exists():
            sections.append(f'<p>Baseline left / recovery right. Green: output; magenta: publisher reference.</p><video controls preload="metadata" src="{video}"></video>')
    if (ROOT / "reserved/summary.json").exists():
        sections.append("<h2>Reserved temporal continuation — all 600 frames</h2>"
                        "<p>Two first-frame visual selections; fresh tracker. Same previously studied match; "
                        "not independent-match validation. Candidate locked before download.</p>")
        for r in json.loads((ROOT / "reserved/summary.json").read_text()):
            sections.append(f"<p>{r['target']} — {r['method']}: {r['correct']}/{r['visible_frames']} correct visible; "
                            f"{r['wrong_person_frames']} wrong-person frames; "
                            f"{r['output_while_absent']} outputs while absent.</p>")
        for folder in sorted((ROOT / "reserved").glob("*_TEST")):
            sections.append(f'<h3>{folder.name}</h3><video controls preload="metadata" src="reserved/{folder.name}/comparison.mp4"></video>')
    (ROOT / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Hockey identity recovery benchmark</title>'
        "<style>body{font:17px system-ui;max-width:1100px;margin:32px auto;padding:16px}"
        "video,img{width:100%}section{margin:48px 0}td,th{padding:8px;text-align:left}"
        "table{border-collapse:collapse}td{border-top:1px solid #ccc}</style>"
        "<h1>Identity recovery: development benchmark</h1>"
        "<p>Seven selected failure cases; four event groups; one previously studied match. "
        "Publisher reference, independent human review pending. This is not a readiness test.</p>"
        "<p>357 case-frame observations reuse 199 unique source frames. "
        "Wrong-person frames are not ID-switch counts. Zero wrong-person frames can coexist "
        "with extensive target loss. Seed binding failures are reported separately.</p>"
        + "".join(sections)
    )
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.name == "artifact_manifest.json":
            continue
        files.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (ROOT / "artifact_manifest.json").write_text(json.dumps(files, indent=2))
    print(f"Built review.html and hashed {len(files)} files")


if __name__ == "__main__":
    main()
