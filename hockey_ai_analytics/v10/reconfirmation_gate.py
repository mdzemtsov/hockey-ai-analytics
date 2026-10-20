"""Experimental safety policy: ambiguous identity cannot self-confirm later."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"v6"))
from gate import ManualGate


class ReconfirmationGate(ManualGate):
    def __init__(self,selections):
        super().__init__(selections)
        self.blocked_reason=None

    def update(self,i,pts,objects):
        if i in self.selections:
            self.blocked_reason=None
        if self.blocked_reason is not None:
            return dict(frame_index=i,pts_s=pts,state="NEEDS_SELECTION",
                        x="",y="",w="",h="",model_track_id="",
                        reason=self.blocked_reason,identity_status="UNCONFIRMED",
                        manual_anchor=False)
        row=super().update(i,pts,objects)
        if row["state"] in ("LOST","AMBIGUOUS","NEEDS_SELECTION"):
            self.blocked_reason="RECONFIRM_AFTER_"+row["reason"]
            row.update(state="NEEDS_SELECTION",reason=self.blocked_reason,
                       identity_status="UNCONFIRMED",x="",y="",w="",h="")
        return row
