"""Check the recorded 50-step SPA trajectory without changing any band list.

This is a fail-closed replay of the current refit, not a replacement selector or
certification of historical inputs. A different trajectory raises an error.
"""
from pathlib import Path
import argparse,csv,hashlib,json,platform,os,contextlib
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'spectral_selection/spa/training_refit/controlled_refit_2026-11-05'
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=REF/'training_spectra_preprocessed.npy'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='caece59bf6c037142e6d80ee5cc11703b69b9d376ccd439362004705f96d8fb0'
    with (REF/'spa50_selection_trace.csv').open(encoding='utf-8-sig') as f:expected=list(csv.DictReader(f))
    matrix=np.load(source,allow_pickle=False)
    assert matrix.shape==(2660,454) and matrix.dtype==np.float32
    trajectory=[];selected=[]
    with contextlib.nullcontext():
        # Match the recorded float32/F-layout numerical path explicitly.
        x=np.asfortranarray(matrix.astype(np.float32)[:,np.arange(10,439)])
        x=x-x.mean(axis=0,keepdims=True,dtype=np.float32)
        std=x.std(axis=0,keepdims=True,ddof=0,dtype=np.float32);std[std<1e-8]=1.
        x=np.asfortranarray(x/std,dtype=np.float32)
        first=int(np.argmax(np.var(x,axis=0,ddof=0,dtype=np.float32)))
        selected.append(first)
        trajectory.append({'step':1,'band_index':first+10,'residual_norm':float(np.linalg.norm(x[:,first]))})
        for step in range(2,51):
            best=None;score=-1.
            for j in range(429):
                if j in selected:continue
                target=x[:,j];s=x[:,selected]
                coef,*_=np.linalg.lstsq(s,target,rcond=None)
                residual=target-s@coef;norm=float(np.linalg.norm(residual))
                if norm>score:best=j;score=norm
            selected.append(best);trajectory.append({'step':step,'band_index':best+10,'residual_norm':score})
        libs=np.show_config(mode='dicts')
    differences=[{'step':r['step'],'recorded':int(e['band_index']),'replayed':r['band_index']} for r,e in zip(trajectory,expected) if r['band_index']!=int(e['band_index'])]
    report={'scope':'current-environment replay only; no new training and no historical provenance certification',
            'python':platform.python_version(),'numpy':np.__version__,'libraries':libs,'thread_control':'single-thread environment variables set before numpy import; runtime worker count not independently measured','input_dtype':str(matrix.dtype),
            'input_C_contiguous':bool(matrix.flags.c_contiguous),'input_F_contiguous':bool(matrix.flags.f_contiguous),
            'standardized_layout':'F-contiguous float32','ddof':0,'initialization':'first exact maximum of standardized variance',
            'residual_tie_rule':'ascending candidate scan; replace only on strict greater-than',
            'full_50_step_indices_match':not differences,'differences':differences,'trajectory':trajectory}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    if differences:raise RuntimeError('Numerical path diverged; output records mismatch. Frozen lists were not changed.')
    print('PASS: all 50 selected indices match the frozen current-refit trajectory; this does not establish universal numerical stability.')
if __name__=='__main__':main()
