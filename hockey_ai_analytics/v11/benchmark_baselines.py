from pathlib import Path
import sys,json,csv,collections
from score_recovery import evaluate,iou
ROOT=Path(__file__).resolve().parent


def write(path,rows):
    with path.open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    protocol=json.loads((ROOT/"protocol.json").read_text());seq=protocol["development_source"]
    summary=[]
    for method in ["baseline","ocsort","deepocsort"]:
        path=ROOT.parent/"v9/results"/seq/"tracks.csv" if method=="baseline" else ROOT.parent/"v10/trackers"/method/seq/"tracks.csv"
        by=collections.defaultdict(list)
        with path.open() as f:
            for r in csv.DictReader(f):
                by[int(r["frame"])].append(dict(id=int(r["id"]),box=[float(r[k]) for k in ("x1","y1","x2","y2")]))
        for case in protocol["cases"]:
            folder=ROOT/"cases"/case["id"];out=ROOT/"results"/method/case["id"];out.mkdir(parents=True,exist_ok=True)
            request=json.loads((folder/"request.json").read_text())
            # Predictions use seed-only request plus cached causal tracker outputs.
            ranked=sorted([(iou(request["seed_bbox_xyxy"],r["box"]),r["id"]) for r in by[request["first_source_frame"]]],reverse=True)
            bound=ranked[0][1] if ranked and ranked[0][0]>=.5 and (len(ranked)==1 or ranked[0][0]-ranked[1][0]>=.1) else None
            predictions=[]
            for frame in range(request["first_source_frame"],request["last_source_frame"]+1):
                r=next((r for r in by[frame] if r["id"]==bound),None)
                predictions.append(r["box"] if r else None)
            (out/"predictions.json").write_text(json.dumps(dict(initial_id=bound,boxes=predictions),indent=2))
            # Reference first read after prediction file is saved.
            with (folder/"reference.csv").open() as f:refs=list(csv.DictReader(f))
            target=int(refs[0]["target_id"])
            reference=[dict(visible=r["visible"]=="True",box=[float(r[k]) for k in ("x1","y1","x2","y2")] if r["visible"]=="True" else None) for r in refs]
            others=collections.defaultdict(list)
            with (folder/"all_reference.csv").open() as f:
                for r in csv.DictReader(f):
                    if int(r["id"])!=target:others[int(r["frame"])].append([float(r[k]) for k in ("x1","y1","x2","y2")])
            rows,m=evaluate(predictions,reference,others,request["fps"])
            m.update(case=case["id"],group=case["group"],method=method,seed_bound=bound is not None)
            write(out/"evaluation.csv",rows);(out/"metrics.json").write_text(json.dumps(m,indent=2));summary.append(m)
    write(ROOT/"baseline_summary.csv",summary)
    (ROOT/"baseline_summary.json").write_text(json.dumps(summary,indent=2))
    for r in summary:print(r["method"],r["case"],r["correct"],r["visible_frames"],r["wrong_person_frames"],r["seed_bound"])


if __name__=="__main__":main()
