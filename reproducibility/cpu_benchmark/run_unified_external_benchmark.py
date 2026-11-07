"""Uniform six-model FP32 CPU inference benchmark; no dataset access or training."""
from pathlib import Path
import argparse,csv,hashlib,json,os,platform,random,statistics,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[2]
ORIGINAL=Path(r'F:\00 Paper-Summary\1.HSMF-Net-Paper\01-Experiments\MobileNet_Ablation_Code-V3')
sys.path[:0]=[str(ROOT/'src'),str(ORIGINAL/'.external_runtime')]
import torch,yaml,numpy as np
from fvcore.nn import FlopCountAnalysis
from mobilenet_enterprise.models.external import build_external_model

METHODS=['cloln','mixersenet','convsst','resnet_rh','mft','hsmf']
OUT=ROOT/'results/statistics/unified_external_benchmark_2026-11-07'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf-8')
def csvsave(p,rows):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def worker(rep):
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(42)
 rgb=torch.randn(1,3,224,224);hsi=torch.rand(1,1,24,24,24)
 order=METHODS.copy();random.Random(20261107+rep).shuffle(order)
 rows=[];details=[]
 for method in order:
  cfg=yaml.safe_load((ROOT/f'configs/external_seed42/{method}_seed42.yaml').read_text('utf-8'))
  model=build_external_model(method,cfg).cpu().eval()
  ckpt=(ROOT/'checkpoints/reference/S8_seed42.pt' if method=='hsmf' else Path(cfg['output_dir'])/cfg['experiment_name']/'checkpoints/best.pt')
  state=torch.load(ckpt,map_location='cpu',weights_only=False)
  stream='model' if method=='hsmf' else ('ema_model' if state.get('ema_eval_active',False) else 'model')
  model.load_state_dict(state[stream],strict=True)
  profiler=FlopCountAnalysis(model,(rgb,hsi)).unsupported_ops_warnings(False).uncalled_modules_warnings(False).tracer_warnings('none')
  macs={k:float(v) for k,v in profiler.by_operator().items() if k in {'conv','linear','einsum','matmul','bmm'}}
  with torch.inference_mode():
   result=model(rgb,hsi);assert result.shape==(1,4) and torch.isfinite(result).all()
   for _ in range(50):model(rgb,hsi)
   for i in range(200):
    start=time.perf_counter_ns();model(rgb,hsi);elapsed=(time.perf_counter_ns()-start)/1e6
    rows.append({'method':method,'repetition':rep,'iteration':i,'latency_ms':elapsed})
  values=[r['latency_ms'] for r in rows if r['method']==method]
  details.append({'method':method,'repetition':rep,'params_m':sum(p.numel() for p in model.parameters())/1e6,'gflops':2*sum(macs.values())/1e9,'macs_by_operator':macs,'mean_ms':statistics.mean(values),'iteration_sd_ms':statistics.stdev(values),'checkpoint':str(ckpt),'checkpoint_sha256':sha(ckpt),'stream':stream})
  print(f'Repetition {rep+1}/5 {method}: {statistics.mean(values):.4f} ms',flush=True)
 csvsave(OUT/f'raw_rep{rep}.csv',rows);save(OUT/f'details_rep{rep}.json',details)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 if args.worker is not None:worker(args.worker);return
 if (OUT/'summary.csv').exists():raise RuntimeError('Completed benchmark exists; preserve its observations.')
 protocol={'date':'2026-11-07','device':'CPU','cpu':'Intel Core i5-14600KF','python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,'threads':4,'interop_threads':1,'precision':'float32','batch_size':1,'rgb_shape':[1,3,224,224],'hsi_shape':[1,1,24,24,24],'weights':'prespecified seed42; external validation-selected stream; S8 ordinary reference weights','warmup_per_model_per_repetition':50,'timed_forwards_per_model_per_repetition':200,'fresh_process_repetitions':5,'latency_report':'mean of five repetition means; sample SD of those five means (ddof=1)','gfops_report':'2 FLOPs per MAC; conv/linear/einsum/matmul/bmm; normalization, activations, interpolation and elementwise operations omitted consistently','timed_scope':'model forward including in-model input adapters; tensors preallocated on CPU; eval and inference_mode; no acquisition/file decoding/preprocessing/transfer','clock':'perf_counter_ns','order':'reproducibly randomized per repetition','benchmark_source_sha256':sha(Path(__file__))}
 save(OUT/'protocol.json',protocol)
 for rep in range(5):
  subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(rep)],check=True)
 details=[r for rep in range(5) for r in json.loads((OUT/f'details_rep{rep}.json').read_text('utf-8'))];summaries=[]
 for m in METHODS:
  rr=[r for r in details if r['method']==m];assert len(rr)==5 and len({r['gflops'] for r in rr})==1
  means=[r['mean_ms'] for r in rr]
  raw=[float(r['latency_ms']) for rep in range(5) for r in csv.DictReader((OUT/f'raw_rep{rep}.csv').open()) if r['method']==m];assert len(raw)==1000
  summaries.append({'method':m,'params_m':rr[0]['params_m'],'gflops':rr[0]['gflops'],'repetitions':5,'timed_forwards':len(raw),'latency_ms_mean':statistics.mean(means),'latency_ms_sd_of_repetition_means':statistics.stdev(means),'pooled_iteration_sd_ms':statistics.stdev(raw),'p50_ms':float(np.percentile(raw,50)),'p95_ms':float(np.percentile(raw,95))})
 csvsave(OUT/'summary.csv',summaries);print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':main()
