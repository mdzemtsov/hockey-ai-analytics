"""Read audited legacy training checkpoint with a restricted global allowlist."""
import pickle,types,collections,torch

def safe_getattr(obj,name):
 if isinstance(obj,torch.optim.lr_scheduler.OneCycleLR) and name in ('_annealing_cos','_annealing_linear'):
  return getattr(obj,name)
 raise pickle.UnpicklingError(f'Unexpected getattr {type(obj)} {name}')
ALLOWED={('collections','OrderedDict'):collections.OrderedDict,('collections','defaultdict'):collections.defaultdict,('torch._utils','_rebuild_tensor_v2'):torch._utils._rebuild_tensor_v2,('torch._utils','_rebuild_parameter'):torch._utils._rebuild_parameter,('torch','FloatStorage'):torch.FloatStorage,('__builtin__','getattr'):safe_getattr,('__builtin__','dict'):dict,('torch.optim.lr_scheduler','OneCycleLR'):torch.optim.lr_scheduler.OneCycleLR,('torch.optim.adamw','AdamW'):torch.optim.AdamW}
class RestrictedUnpickler(pickle.Unpickler):
 def find_class(self,module,name):
  if (module,name) not in ALLOWED:raise pickle.UnpicklingError(f'Forbidden global {module}.{name}')
  return ALLOWED[module,name]
def load_checkpoint(path):
 module=types.SimpleNamespace(__name__='restricted_checkpoint_pickle',Unpickler=RestrictedUnpickler,load=pickle.load)
 return torch.load(path,map_location='cpu',pickle_module=module,weights_only=False)
if __name__=='__main__':
 from pathlib import Path
 r=Path(__file__).resolve().parent;x=load_checkpoint(r/'models/hockey_parseq.ckpt');print(x['hyper_parameters']);torch.save(dict(state_dict=x['state_dict'],hyper_parameters=x['hyper_parameters']),r/'models/hockey_parseq_inference.pt')
 s=torch.load(r/'models/hockey_legibility.pth',map_location='cpu',weights_only=True);print('Legibility keys',list(s)[:5])
