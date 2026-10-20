"""Fixed-schedule supervised contrastive training; no test images or labels."""
from pathlib import Path
import json,random
import numpy as np
import torch
from metric_model import MetricHead,objective

R=Path(__file__).resolve().parent


def main():
    cfg=json.loads((R/"protocol.json").read_text());seed=cfg["seed"]
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    rows=json.loads((R/"features.json").read_text())
    features=np.load(R/"features.npz")["features"]
    identities=sorted({r["identity"] for r in rows if r["split"]=="train"})
    val_ids={r["identity"] for r in rows if r["split"]=="validation"}
    assert not set(identities)&val_ids
    assert all(not r["sequence"].startswith("CHI_") for r in rows)
    groups={identity:[i for i,r in enumerate(rows) if r["identity"]==identity] for identity in identities}
    team_names=sorted({r["team"] for r in rows})
    model=MetricHead();optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=1e-4)
    rng=np.random.default_rng(seed);history=[]
    for epoch in range(cfg["epochs"]):
        for _ in range(cfg["steps_per_epoch"]):
            indices=np.concatenate([rng.choice(groups[identity],4,replace=False) for identity in identities])
            x=torch.tensor(features[indices])
            labels=torch.tensor([identities.index(rows[i]["identity"]) for i in indices])
            teams=torch.tensor([team_names.index(rows[i]["team"]) for i in indices])
            optimizer.zero_grad()
            loss,terms=objective(model(x),labels,teams,cfg["temperature"],cfg["triplet_margin"])
            loss.backward();optimizer.step()
        history.append(dict(epoch=epoch+1,loss=float(loss.detach()),**terms))
    torch.save(model.state_dict(),R/"hockey_metric_head.pt")
    (R/"training_history.json").write_text(json.dumps(history,indent=2))
    (R/"training_summary.json").write_text(json.dumps(dict(train_identities=identities,validation_identities=sorted(val_ids),
        train_crops=sum(r["split"]=="train" for r in rows),validation_crops=sum(r["split"]=="validation" for r in rows),
        seed=seed,epochs=cfg["epochs"],backbone_frozen=True,parameters_trained=sum(p.numel() for p in model.parameters()),
        final_loss=history[-1]),indent=2))
    print((R/"training_summary.json").read_text())


if __name__=="__main__":main()
