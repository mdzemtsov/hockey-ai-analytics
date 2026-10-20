"""Evaluation only; labels must never be supplied to inference."""
import numpy as np


def iou(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(inter/max(1e-9,(a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-inter))


def evaluate(predictions,reference,others,fps=30):
    if len(predictions)!=len(reference):raise ValueError("Missing frames may not be dropped")
    states=[];rows=[]
    for index,(pred,ref) in enumerate(zip(predictions,reference),1):
        visible=bool(ref["visible"])
        overlap=iou(pred,ref["box"]) if pred is not None and visible else None
        state=("CORRECT" if overlap>=.5 else "WRONG_OR_MISALIGNED") if overlap is not None else (
            "OUTPUT_WHILE_ABSENT" if pred is not None else ("VISIBLE_MISS" if visible else "ABSENT"))
        wrong=pred is not None and state!="CORRECT" and any(iou(pred,b)>=.5 for b in others[index])
        rows.append(dict(frame=index,state=state,iou=overlap,wrong_person=bool(wrong)))
        states.append(state)
    first_failure=next((i for i,s in enumerate(states) if s in ("VISIBLE_MISS","WRONG_OR_MISALIGNED","OUTPUT_WHILE_ABSENT")),None)
    recovery=None
    if first_failure is not None:
        recovery=next((i for i in range(first_failure+1,len(states)-2) if states[i:i+3]==["CORRECT"]*3),None)
    visible=sum(bool(r["visible"]) for r in reference)
    metrics=dict(frames=len(reference),visible_frames=visible,
                 correct=sum(s=="CORRECT" for s in states),
                 visible_misses=sum(s=="VISIBLE_MISS" for s in states),
                 wrong_or_misaligned=sum(s=="WRONG_OR_MISALIGNED" for s in states),
                 output_while_absent=sum(s=="OUTPUT_WHILE_ABSENT" for s in states),
                 wrong_person_frames=sum(r["wrong_person"] for r in rows),
                 correct_visible_coverage=sum(s=="CORRECT" for s in states)/visible if visible else None,
                 first_failure_frame=first_failure+1 if first_failure is not None else None,
                 first_stable_recovery_frame=recovery+1 if recovery is not None else None,
                 recovery_delay_s=(recovery-first_failure)/fps if recovery is not None else None,
                 recovery_censored=first_failure is not None and recovery is None,
                 human_confirmations_used=0)
    return rows,metrics
