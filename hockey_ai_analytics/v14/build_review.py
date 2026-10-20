"""Expose every fitted/calibration crop for review; never auto-confirm labels."""
from pathlib import Path
import collections,csv,hashlib,html,json
import cv2
R=Path(__file__).resolve().parent


def main():
    rows=json.loads((R/"features.json").read_text());groups=collections.defaultdict(list)
    out=R/"review_crops";out.mkdir(exist_ok=True)
    last_key=None;image=None
    for i,row in enumerate(rows):
        key=(row["sequence"],row["frame"])
        if key!=last_key:
            image=cv2.imread(str(R/"data"/row["sequence"]/"img1"/f"{row['frame']:06d}.jpg"));last_key=key
        x1,y1,x2,y2=map(int,row["box"])
        crop=cv2.resize(image[y1:y2,x1:x2],(128,160))
        path=out/f"{i:04d}.jpg";cv2.imwrite(str(path),crop)
        groups[row["identity"]].append((i,row,str(path.relative_to(R))))
    sections=[]
    for identity,items in groups.items():
        sid=html.escape(identity)
        photos="".join(f'<label><img loading="lazy" src="{path}"><br>Frame {row["frame"]}<br>'
                      f'<input class="exclude" data-index="{i}" type="checkbox"> Exclude crop</label>' for i,row,path in items)
        sections.append(f'<details data-identity="{sid}"><summary>{sid} — {items[0][1]["team"]} — {items[0][1]["split"]} — {len(items)} crops</summary>'
                        '<p>Review every crop before recording consistency. Publisher IDs are not personal names.</p>'
                        '<select class="status"><option value="pending">Pending</option><option value="consistent">Reviewed: remaining crops show one player</option>'
                        '<option value="inconsistent">Different players / correction needed</option><option value="uncertain">Identity uncertain</option></select>'
                        '<input class="note" placeholder="Notes / visible jersey evidence"><div class="grid">'+photos+'</div></details>')
    digest=hashlib.sha256((R/"features.json").read_bytes()).hexdigest()
    script="""
document.querySelector('#export').onclick=()=>{
 const reviewer=document.querySelector('#reviewer').value.trim();
 if(!reviewer){document.querySelector('#message').textContent='Enter the reviewer name before exporting.';return;}
 const identities=[...document.querySelectorAll('details')].map(d=>({
  identity:d.dataset.identity,status:d.querySelector('.status').value,
  note:d.querySelector('.note').value,
  excluded_crop_indices:[...d.querySelectorAll('.exclude:checked')].map(x=>Number(x.dataset.index))
 }));
 const result={schema:'hockey_identity_review_v1',dataset_sha256:DATASET_HASH,reviewer,
  submitted_at:new Date().toISOString(),source:'manual_review_ui',identities};
 const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='identity_review.json';a.click();
 setTimeout(()=>URL.revokeObjectURL(a.href),1000);
 document.querySelector('#message').textContent='Exported review overlay. Source labels and model have not been changed.';
};
"""
    (R/"review_labels.html").write_text('<!doctype html><meta charset="utf-8"><title>Review hockey identity examples</title>'
        '<style>body{font:16px system-ui;max-width:1300px;margin:25px auto;padding:15px}.grid{display:flex;flex-wrap:wrap;gap:12px}.grid label{width:132px}details{padding:15px;border:1px solid #aaa;margin:12px 0}input,select,button{padding:8px;margin:6px}summary{cursor:pointer}</style>'
        '<h1>Training identity review — all examples pending independent review</h1>'
        '<p>This page records a separate review overlay. It does not change publisher annotations or retrain the model. '
        'Reviewers may exclude individual crops and flag mixed identities. No box is pre-approved. Unsaved choices are lost on closing; export to preserve them.</p>'
        '<input id="reviewer" placeholder="Reviewer name"><button id="export">Export review JSON</button><p id="message"></p>'
        +''.join(sections)+f'<script>const DATASET_HASH={json.dumps(digest)};'+script+'</script>')
    results=json.loads((R/"test/summary.json").read_text());tr=[]
    for name,s in results.items():
        tr.append(f"<tr><td>{name}</td><td>{s['top1_correct']}/{s['known_queries']}</td><td>{s['same_team_errors']}</td>"
                  f"<td>{s['accepted_known']}</td><td>{s['accepted_unknown']}</td><td>{s['accepted_correct']}/{s['accepted_total']}</td></tr>")
    (R/"review.html").write_text('<!doctype html><meta charset="utf-8"><title>Hockey-specific training results</title>'
        '<style>body{font:17px system-ui;max-width:1100px;margin:35px auto;padding:20px}td,th{padding:12px;text-align:left}</style>'
        '<h1>Hockey-specific identity training — expanded experiment</h1>'
        '<p>Trained metric head; frozen general backbone. This is publisher-box retrieval with labelled enrollment, not working player tracking.</p>'
        '<table><tr><th>Variant / max views</th><th>Correct / known queries</th><th>Same-team errors</th>'
        '<th>Accepted known</th><th>Accepted unknown</th><th>Correct / all accepted</th></tr>'+''.join(tr)+'</table>'
        '<p><a href="QA.md">QA and limitations</a> · <a href="review_labels.html">Review every training/calibration crop</a> · '
        '<a href="hockey_metric_head.pt">Trained head weights</a></p>'
        '<p>Zero accepted matches is total abstention, not perfect identity recognition. Test references are unchanged publisher labels, not independently human-verified GT.</p>')
    print("Saved review pages and",len(rows),"crop images")


if __name__=="__main__":main()
