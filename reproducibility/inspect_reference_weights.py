"""Reference weight-load check; synthetic tensors only, no research-data accuracy."""
import argparse,json,sys
from pathlib import Path
import torch,yaml
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from mobilenet_enterprise.registry import build_model
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--stream',required=True,choices=['model','ema_model'])
a=p.parse_args()
torch.set_num_threads(4)
for label in ['S7','S8']:
    cfg=yaml.safe_load((root/'configs/archived'/f'{label}_seed42.yaml').read_text(encoding='utf-8'))
    cfg['pretrained']={'enabled':False,'source':'none: reference state follows'}
    model=build_model(cfg).cpu().eval()
    state=torch.load(root/'checkpoints/reference'/f'{label}_seed42.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state[a.stream],strict=True)
    torch.manual_seed(42)
    with torch.inference_mode():
        y=model(torch.randn(1,3,224,224),torch.randn(1,1,cfg['data']['hsi_bands'],24,24))
    assert list(y.shape)==[1,4] and torch.isfinite(y).all()
    print(json.dumps({'model':label,'stream':a.stream,'strict_load':True,'synthetic_shape':[1,4],'historical_accuracy_reproduced':False}))
