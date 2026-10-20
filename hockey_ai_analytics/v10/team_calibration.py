"""Explicit two-team exemplars. No unsupervised assumption that two teams are visible."""
from appearance_tracker import descriptor, distance


class TeamCalibration:
    def __init__(self, min_separation=.25, max_distance=.60, min_margin=.12):
        self.examples={}
        self.min_separation=min_separation
        self.max_distance=max_distance
        self.min_margin=min_margin

    def add(self, label, image, box):
        if label not in ("TEAM_A","TEAM_B"):
            raise ValueError("Explicit TEAM_A or TEAM_B required")
        feat=descriptor(image,box)
        if feat is None:
            raise ValueError("Exemplar too small or outside image")
        self.examples[label]=feat

    def readiness(self):
        if len(self.examples)!=2:
            return "NEEDS_BOTH_TEAMS"
        if distance(self.examples["TEAM_A"],self.examples["TEAM_B"])<self.min_separation:
            return "EXAMPLES_TOO_SIMILAR"
        return "READY_FOR_HYPOTHESES"

    def classify(self,image,box,object_class=4):
        if object_class==6:
            return dict(team="NOT_APPLICABLE",reason="REFEREE",margin=None)
        state=self.readiness()
        if state!="READY_FOR_HYPOTHESES":
            return dict(team="UNKNOWN",reason=state,margin=None)
        feat=descriptor(image,box)
        if feat is None:
            return dict(team="UNKNOWN",reason="INSUFFICIENT_PIXELS",margin=None)
        ranks=sorted((distance(feat,v),k) for k,v in self.examples.items())
        best,label=ranks[0];margin=ranks[1][0]-best
        if best>self.max_distance or margin<self.min_margin:
            return dict(team="UNKNOWN",reason="AMBIGUOUS_APPEARANCE",margin=margin)
        return dict(team=label,reason="APPEARANCE_HYPOTHESIS",margin=margin)
