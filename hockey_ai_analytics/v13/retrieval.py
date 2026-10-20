"""Labelled enrollment evaluation, not an end-to-end tracker."""
from pathlib import Path
import json,csv
import numpy as np
import torch
from metric_model import MetricHead

R=Path(__file__).resolve().parent


def project(features,trained):
    if not trained:return features
    model=MetricHead()
    model.load_state_dict(torch.load(R/"hockey_metric_head.pt",map_location="cpu",weights_only=True))
    model.eval()
    with torch.inference_mode():return model(torch.tensor(features,dtype=torch.float32)).numpy()


def rank(gallery,gallery_features,queries,query_features,views):
    records=[]
    for row,feature in zip(queries,query_features):
        ids=sorted({r["identity"] for r in gallery if r["sequence"]==row["sequence"]})
        candidates=[]
        for identity in ids:
            indices=[i for i,r in enumerate(gallery) if r["identity"]==identity][:views]
            sims=gallery_features[indices]@feature
            index=indices[int(np.argmax(sims))]
            candidates.append((float(max(sims)),identity,gallery[index]["team"]))
        candidates.sort(reverse=True)
        best,predicted,team=candidates[0]
        margin=best-candidates[1][0] if len(candidates)>1 else 0.
        known=row["identity"] in ids
        correct=predicted==row["identity"]
        records.append(dict(sequence=row["sequence"],frame=row["frame"],identity=row["identity"],
                            predicted=predicted,known=known,correct=correct,
                            same_team_error=known and not correct and team==row["team"],
                            similarity=best,margin=margin))
    return records


def summarize(rows,threshold):
    known=[r for r in rows if r["known"]];unknown=[r for r in rows if not r["known"]]
    accepted=[r for r in rows if r["margin"]>=threshold]
    known_accepted=[r for r in accepted if r["known"]]
    return dict(known_queries=len(known),unknown_personnel_queries=len(unknown),
        top1_correct=sum(r["correct"] for r in known),
        top1_accuracy=sum(r["correct"] for r in known)/len(known) if known else None,
        same_team_errors=sum(r["same_team_error"] for r in known),
        threshold=threshold,accepted_known=len(known_accepted),
        accepted_known_correct=sum(r["correct"] for r in known_accepted),
        accepted_unknown=sum(not r["known"] for r in accepted),
        accepted_total=len(accepted),accepted_correct=sum(r["correct"] for r in accepted),
        accepted_precision=sum(r["correct"] for r in accepted)/len(accepted) if accepted else None,
        known_acceptance_coverage=len(known_accepted)/len(known) if known else None)


def save_rows(path,rows):
    with path.open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def calibrate():
    rows=json.loads((R/"features.json").read_text());features=np.load(R/"features.npz")["features"]
    gi=[];qi=[]
    for identity in sorted({r["identity"] for r in rows}):
        indices=[i for i,r in enumerate(rows) if r["identity"]==identity]
        half=len(indices)//2
        gi += [indices[i] for i in np.linspace(0,half-1,min(4,half)).astype(int)]
        if rows[indices[0]]["split"]=="validation":qi+=indices[half:]
    assert not set(gi)&set(qi)
    assert all(rows[i]["split"]=="validation" for i in qi)
    results={}
    for trained in (False,True):
        emb=project(features,trained)
        for views in (1,4):
            name=f"{'trained' if trained else 'generic'}_{views}"
            ranked=rank([rows[i] for i in gi],emb[gi],[rows[i] for i in qi],emb[qi],views)
            threshold=max([.03]+[r["margin"]+1e-6 for r in ranked if not r["correct"]])
            results[name]=summarize(ranked,threshold)
            save_rows(R/f"calibration_{name}.csv",ranked)
    (R/"calibration.json").write_text(json.dumps(results,indent=2))
    (R/"calibration_split.json").write_text(json.dumps(dict(gallery_indices=gi,query_indices=qi),indent=2))
    print(json.dumps(results,indent=2))


if __name__=="__main__":calibrate()
