"""Exploratory hockey-tuned legibility + PARSeq; outputs are unverified number hypotheses."""
from pathlib import Path
import sys,json,collections
import torch
from torch import nn
from torchvision import models,transforms as T
from PIL import Image
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/jersey-number-pipeline/str/parseq'))
from strhub.models.parseq.system import PARSeq

torch.set_num_threads(4)
checkpoint=torch.load(ROOT/'models/hockey_parseq_inference.pt',map_location='cpu',weights_only=True);model=PARSeq(**checkpoint['hyper_parameters']);model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
class Legibility(nn.Module):
 def __init__(self):
  super().__init__();self.model_ft=models.resnet34(weights=None);self.model_ft.fc=nn.Linear(512,1)
 def forward(self,x):return self.model_ft(x).sigmoid()
leg=Legibility();leg.load_state_dict(torch.load(ROOT/'models/hockey_legibility.pth',map_location='cpu',weights_only=True));leg.eval()
lt=T.Compose([T.Resize((256,256)),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])]);nt=T.Compose([T.Resize(tuple(checkpoint['hyper_parameters']['img_size']),T.InterpolationMode.BICUBIC),T.ToTensor(),T.Normalize(.5,.5)])
items=json.loads((ROOT/'ocr/inputs.json').read_text());out=[]
with torch.inference_mode():
 for item in items:
  im=Image.open(item['path']).convert('RGB');readability=float(leg(lt(im).unsqueeze(0)).item());probs=model(nt(im).unsqueeze(0))[:,:3,:11].softmax(-1);labels,conf=model.tokenizer.decode(probs);values=conf[0].reshape(-1).tolist();confidence=float(min(values)) if values else 0.;label=labels[0];candidate=label if label.isdigit() and 1<=len(label)<=2 and readability>=.5 and confidence>=.7 else None
  out.append({**item,'legibility_score':readability,'raw_number':label,'min_token_score':confidence,'candidate':candidate,'status':'UNVERIFIED_NUMBER_HYPOTHESIS' if candidate else 'ABSTAIN'})
(ROOT/'ocr/specialized_results.json').write_text(json.dumps(out,indent=2));summary={}
for seq in sorted({x['sequence'] for x in out}):
 rows=[x for x in out if x['sequence']==seq];counts=collections.Counter(x['candidate'] for x in rows if x['candidate']);top=counts.most_common(1);stable=top[0][0] if top and top[0][1]>=3 and top[0][1]/sum(counts.values())>=.75 else None;summary[seq]=dict(sampled=len(rows),accepted_hypotheses=sum(counts.values()),counts=dict(counts),stable_number_hypothesis=stable,identity_confirmation=False)
(ROOT/'ocr/specialized_summary.json').write_text(json.dumps(summary,indent=2));print(summary)
