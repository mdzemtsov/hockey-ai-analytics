"""Evidence-only overlays. Publisher boxes are never fed back to inference."""
from pathlib import Path
import csv
import json
import subprocess
import cv2
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent


def render(folder, image_dir, baseline, recovery, output):
    refs = list(csv.DictReader((folder / "reference.csv").open()))
    request = json.loads((folder / "request.json").read_text())
    assert len(baseline) == len(recovery) == len(refs)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1280x400",
           "-r", str(request["fps"]), "-i", "-", "-an", "-c:v", "libx264",
           "-crf", "20", "-pix_fmt", "yuv420p", str(output)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for index, (old, new, ref) in enumerate(zip(baseline, recovery, refs)):
        frame = request["first_source_frame"] + index
        source = cv2.imread(str(image_dir / f"{frame:06d}.jpg"))
        panels = []
        for title, prediction in [("Baseline fixed ID", old), ("Recovery v1", new)]:
            im = source.copy()
            if ref["visible"] == "True":
                b = [round(float(ref[k])) for k in ("x1", "y1", "x2", "y2")]
                cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), (255, 0, 255), 2)
            b = prediction["box"]
            if b is not None:
                b = [round(v) for v in b]
                cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), (0, 220, 0), 3)
            im = cv2.resize(im, (640, 360))
            im = cv2.copyMakeBorder(im, 40, 0, 0, 0, cv2.BORDER_CONSTANT)
            label = prediction.get("state", "OUTPUT" if b is not None else "UNKNOWN")
            cv2.putText(im, f"{title} | {label} | {frame}", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
            panels.append(im)
        proc.stdin.write(cv2.hconcat(panels).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    cap = cv2.VideoCapture(str(output))
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    assert count == len(refs)
    csv_rows = []
    for prediction in recovery:
        box = prediction["box"]
        csv_rows.append(dict(source_frame=prediction["source_frame"],
            state=prediction["state"], candidate_id=prediction.get("candidate_id"),
            x1=box[0] if box else "", y1=box[1] if box else "",
            x2=box[2] if box else "", y2=box[3] if box else ""))
    with output.with_suffix(".csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    return dict(path=str(output.relative_to(ROOT)), decoded_frames=count)


def main():
    records = []
    for path in sorted((ROOT / "cases").glob("*/request.json")):
        name = path.parent.name
        old = json.loads((ROOT / "results/baseline" / name / "predictions.json").read_text())["boxes"]
        new = json.loads((ROOT / "results/recovery" / name / "predictions.json").read_text())
        records.append(render(path.parent, ROOT.parent / "v9/data/CHI_VS_TOR_004/img1",
                              [dict(box=b) for b in old], new,
                              ROOT / "results/recovery" / name / "comparison.mp4"))
    if (ROOT / "reserved/summary.json").exists():
        for folder in sorted((ROOT / "reserved").glob("*_TEST")):
            old = json.loads((folder / "baseline/predictions.json").read_text())
            new = json.loads((folder / "recovery/predictions.json").read_text())
            records.append(render(folder, ROOT / "reserved/img1", old, new, folder / "comparison.mp4"))
    (ROOT / "recovery_video_validation.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
