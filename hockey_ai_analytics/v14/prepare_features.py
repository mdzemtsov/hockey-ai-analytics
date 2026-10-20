"""Extract immutable image features; keep training and gallery manifests separate."""
from pathlib import Path
import sys,json,hashlib,collections
import cv2,numpy as np
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/"v12"))
from evidence import Encoder


def extract(rows,out):
    encoder=Encoder();by=collections.defaultdict(list)
    for i,row in enumerate(rows):by[(row["sequence"],row["frame"])].append((i,row))
    features=np.zeros((len(rows),384),np.float32);sources=[]
    for (seq,frame),items in by.items():
        path=(R.parent/"v9/data" if seq=="CHI_VS_TOR_004" else R/"data")/seq/"img1"/f"{frame:06d}.jpg"
        image=cv2.imread(str(path));assert image is not None
        embeddings=encoder.encode(image,[row["box"] for _,row in items])
        for (index,row),emb in zip(items,embeddings):
            assert emb is not None
            features[index]=emb
        sources.append(dict(path=str(path.relative_to(R.parent)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    np.savez_compressed(R/f"{out}.npz",features=features)
    (R/f"{out}.json").write_text(json.dumps(rows,indent=2))
    (R/f"{out}_sources.json").write_text(json.dumps(sources,indent=2))


def main():
    labels=json.loads((R/"audit_labels.json").read_text());p=json.loads((R/"protocol.json").read_text())
    by=collections.defaultdict(list)
    for row in json.loads((R/"candidate_crops.json").read_text()):
        if str(row["id"]) in labels[row["sequence"]]["teams"]:by[(row["sequence"],row["id"])].append(row)
    rows=[]
    for (seq,identity),samples in by.items():
        indices=np.linspace(0,len(samples)-1,min(p["max_crops_per_identity"],len(samples))).astype(int)
        for i in indices:
            rows.append(dict(**samples[i],identity=f"{seq}:{identity}",team=f"{seq}:{labels[seq]['teams'][str(identity)]}",
                             split="validation" if identity in labels[seq]["validation_ids"] else "train",
                             reference="publisher_with_assistant_sample_review",human_verified=False))
    extract(rows,"features")
    gallery=json.loads((R/"gallery_labels.json").read_text());by=collections.defaultdict(list)
    for row in json.loads((R/"gallery_candidates.json").read_text()):
        if str(row["id"]) in gallery["teams"]:by[row["id"]].append(row)
    rows=[]
    for identity,samples in by.items():
        for i in np.linspace(0,len(samples)-1,min(12,len(samples))).astype(int):
            row=samples[i]
            rows.append(dict(**row,identity=f"{row['sequence']}:{identity}",team=f"{row['sequence']}:{gallery['teams'][str(identity)]}",
                             split="test_gallery",reference="publisher_enrollment_surrogate",human_verified=False))
    extract(rows,"gallery")
    print("Saved train/validation features and separate test enrollment features",flush=True)


if __name__=="__main__":main()
