"""Hockey-specific metric head; frozen generic backbone is external."""
import torch
from torch import nn
import torch.nn.functional as F


class MetricHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(384,256),nn.ReLU(),nn.Linear(256,128))

    def forward(self,x):
        return F.normalize(self.net(x),dim=-1)


def objective(z,labels,teams,temperature=.1,margin=.2):
    similarity=z@z.T
    eye=torch.eye(len(z),device=z.device,dtype=torch.bool)
    positives=(labels[:,None]==labels[None,:])&~eye
    negatives=(labels[:,None]!=labels[None,:])
    hard_team=negatives&(teams[:,None]==teams[None,:])
    logits=(similarity/temperature).masked_fill(eye,-1e4)
    logp=logits-torch.logsumexp(logits,dim=1,keepdim=True)
    counts=positives.sum(1)
    valid=counts>0
    supcon=(-(logp*positives).sum(1)/counts.clamp_min(1))[valid].mean()
    # Farthest positive and closest same-team negative: explicitly train against
    # uniform-only matching. Anchors without a same-team negative skip this term.
    worst_positive=similarity.masked_fill(~positives,2).min(1).values
    best_negative=similarity.masked_fill(~hard_team,-2).max(1).values
    eligible=valid&hard_team.any(1)
    triplet=F.relu(margin+best_negative-worst_positive)[eligible].mean() if eligible.any() else z.sum()*0
    return supcon+.5*triplet,dict(supcon=float(supcon.detach()),same_team_triplet=float(triplet.detach()),
                                same_team_anchors=int(eligible.sum()))
