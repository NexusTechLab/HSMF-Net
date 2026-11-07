"""Exact two-sided signed-rank follow-up; preserve original prediction records."""
from pathlib import Path
import csv,itertools,json,statistics,hashlib
import numpy as np
from scipy.stats import rankdata,t
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/statistics/two_sided'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def pvalue(d):
    nz=[v for v in d if v]
    if not nz:return 1.
    ranks=rankdata([abs(v) for v in nz],method='average')
    observed=sum(r for v,r in zip(nz,ranks) if v>0)
    possible=[sum(r for positive,r in zip(signs,ranks) if positive) for signs in itertools.product([False,True],repeat=len(nz))]
    return min(1.,2*min(sum(v<=observed for v in possible),sum(v>=observed for v in possible))/len(possible))
def family(source,variant_key,full,baselines,name):
    rows=read(source);lookup={(r[variant_key],int(r['seed'])):r for r in rows};summary=[];seedrows=[]
    for b in baselines:
        integer=[int(lookup[full,s]['correct'])-int(lookup[b,s]['correct']) for s in range(42,47)]
        d=[100*v/570 for v in integer];mean=statistics.mean(d);sd=statistics.stdev(d);half=float(t.ppf(.975,4))*sd/np.sqrt(5)
        summary.append(dict(contrast=f'{full} vs {b}',n_pairs=5,mean_accuracy_difference_pp=mean,ci95_lower_pp=mean-half,ci95_upper_pp=mean+half,two_sided_signed_rank_p=pvalue(integer),holm_adjusted_p=0.))
        for s,v,pp in zip(range(42,47),integer,d):seedrows.append(dict(contrast=f'{full} vs {b}',seed=s,correct_difference=v,accuracy_difference_pp=pp))
    previous=0.
    for k,i in enumerate(sorted(range(len(summary)),key=lambda i:summary[i]['two_sided_signed_rank_p'])):
        previous=max(previous,min(1.,(len(summary)-k)*summary[i]['two_sided_signed_rank_p']))
        summary[i]['holm_adjusted_p']=previous
    for file,rr in [(f'{name}_contrasts.csv',summary),(f'{name}_per_seed_differences.csv',seedrows)]:
        with (OUT/file).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    return summary
def main():
    OUT.mkdir(exist_ok=True)
    historical=family(ROOT/'results/statistics/per_seed_metrics.csv','model','S8',[f'S{i}' for i in range(1,8)],'historical')
    matched=family(ROOT/'results/statistics/matched24/per_seed_metrics.csv','variant','full24',['concat24','gated24','cross_attention24','no_modulation24'],'matched24')
    report={'date':'2026-11-06','analysis_status':'exploratory revision analysis; no recovered prespecified directional hypothesis',
        'method':'two-sided exact sign enumeration of nonzero correct-count differences; average absolute ranks; p=min(1,2*min(P(W+<=observed),P(W+>=observed))); zeros omitted',
        'families':'Holm separately for seven historical accuracy contrasts and four matched24 accuracy contrasts',
        'intervals':'unadjusted paired two-sided 95% t intervals, df=4; not simultaneous or inversions of corrected rank tests',
        'predictions_changed':False,'historical':historical,'matched24':matched}
    (OUT/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'historical_Holm':[r['holm_adjusted_p'] for r in historical],'matched24':matched},indent=2))
if __name__=='__main__':main()
