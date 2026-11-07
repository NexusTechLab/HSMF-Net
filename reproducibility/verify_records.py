"""Recalculate metrics from frozen prediction records; Python standard library only.

Run: python verify_records.py
No training, image loading, checkpoint loading, or network access is performed.
"""
from pathlib import Path
import csv, json, math, statistics, hashlib, itertools

ROOT=Path(__file__).resolve().parents[1]
CLASSES=['CX186','XK129','ZM22','ZM379']
SUPPORT=[135,150,150,135]
T_CRITICAL_DF4=2.7764451051977987

def write_csv(name,rows):
    path=ROOT/'results/statistics'/name;path.parent.mkdir(exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summary(values):
    mean=statistics.mean(values);sd=statistics.stdev(values)
    h=T_CRITICAL_DF4*sd/math.sqrt(5)
    return {'mean':mean,'sd':sd,'ci95_lower':mean-h,'ci95_upper':mean+h}

def ranks(values):
    result=[0.0]*len(values);order=sorted(range(len(values)),key=values.__getitem__)
    i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:j+=1
        rank=(i+1+j)/2
        for k in order[i:j]:result[k]=rank
        i=j
    return result

def signed_rank(differences):
    d=[x for x in differences if x!=0];r=ranks([abs(x) for x in d])
    positive=sum(a for a,x in zip(r,d) if x>0);total=sum(r)
    possible=[sum(a for a,sign in zip(r,signs) if sign) for signs in itertools.product([0,1],repeat=len(r))]
    p=sum(x>=positive-1e-12 for x in possible)/len(possible)
    return p,(2*positive-total)/total if total else 0.0

def main():
    manifest=json.loads((ROOT/'reproducibility/records_manifest.json').read_text('utf-8'))
    runs=[];perclass=[];cells=[];matrices={}
    for item in manifest:
        p=ROOT/item['record_file'];raw=p.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==item['record_sha256'],p.name
        rows=list(csv.DictReader(p.open(encoding='utf-8')))
        assert len(rows)==570 and len({x['sample_index'] for x in rows})==570
        cm=[[0]*4 for _ in range(4)]
        for row in rows:cm[int(row['true_label'])][int(row['pred_label'])]+=1
        assert [sum(row) for row in cm]==SUPPORT
        c=sum(cm[i][i] for i in range(4));f1=[]
        for i,name in enumerate(CLASSES):
            tp=cm[i][i];fp=sum(row[i] for row in cm)-tp;fn=sum(cm[i])-tp
            precision=100*tp/(tp+fp) if tp+fp else 0
            recall=100*tp/(tp+fn) if tp+fn else 0
            f=200*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0;f1.append(f)
            perclass.append({'model':item['model'],'seed':item['seed'],'class':name,'support':SUPPORT[i],'precision_pct':precision,'recall_pct':recall,'f1_pct':f})
            for j,pred in enumerate(CLASSES):cells.append({'model':item['model'],'seed':item['seed'],'true_class':name,'pred_class':pred,'count':cm[i][j],'row_pct':100*cm[i][j]/SUPPORT[i]})
        runs.append({'model':item['model'],'seed':item['seed'],'n_test':570,'correct':c,'accuracy_pct':100*c/570,'macro_f1_pct':statistics.mean(f1)})
        matrices[(item['model'],item['seed'])]=cm
    assert len(runs)==40
    aggregate=[];classmean=[];mean_cells=[];plot=[]
    for m in [f'S{i}' for i in range(1,9)]:
        rr=sorted([x for x in runs if x['model']==m],key=lambda x:x['seed'])
        assert [x['seed'] for x in rr]==list(range(42,47))
        row={'model':m,'n_runs':5,'n_test_per_run':570}
        for metric in ['accuracy_pct','macro_f1_pct']:
            row.update({metric+'_'+k:v for k,v in summary([x[metric] for x in rr]).items()})
        aggregate.append(row)
        for c in CLASSES:
            cr=[x for x in perclass if x['model']==m and x['class']==c]
            classmean.append({'model':m,'class':c,'support_per_run':SUPPORT[CLASSES.index(c)],**{k:statistics.mean(x[k] for x in cr) for k in ['precision_pct','recall_pct','f1_pct']}})
        mat=[]
        for i,tc in enumerate(CLASSES):
            mat.append([])
            for j,pc in enumerate(CLASSES):
                count=statistics.mean(matrices[(m,seed)][i][j] for seed in range(42,47));pct=100*count/SUPPORT[i];mat[-1].append(pct)
                mean_cells.append({'model':m,'true_class':tc,'pred_class':pc,'mean_count_per_run':count,'support_per_run':SUPPORT[i],'mean_row_pct':pct})
        acc_from_matrix=sum(SUPPORT[i]*mat[i][i] for i in range(4))/570
        assert abs(acc_from_matrix-row['accuracy_pct_mean'])<1e-10
        plot.append({'model':m,'accuracy':row['accuracy_pct_mean'],'matrix_pct':mat,'support':SUPPORT})
    paired=[]
    for m in [f'S{i}' for i in range(1,8)]:
        d=[next(x['correct'] for x in runs if x['model']=='S8' and x['seed']==seed)-next(x['correct'] for x in runs if x['model']==m and x['seed']==seed) for seed in range(42,47)]
        p,rb=signed_rank(d);st=summary([100*x/570 for x in d])
        paired.append({'contrast':'S8 vs '+m,'n_seed_pairs':5,'mean_difference_pp':st['mean'],'sd_difference_pp':st['sd'],'ci95_lower_pp':st['ci95_lower'],'ci95_upper_pp':st['ci95_upper'],'one_sided_signflip_signed_rank_p':p,'holm_p':0.0,'rank_biserial':rb})
    order=sorted(range(7),key=lambda i:paired[i]['one_sided_signflip_signed_rank_p']);last=0
    for rank,i in enumerate(order):
        last=max(last,min(1,(7-rank)*paired[i]['one_sided_signflip_signed_rank_p']));paired[i]['holm_p']=last
    rank_sums=[0.0]*8;ties=0
    for seed in range(42,47):
        vals=[next(x['correct'] for x in runs if x['model']==m and x['seed']==seed) for m in [f'S{i}' for i in range(1,9)]]
        rr=ranks(vals);rank_sums=[a+b for a,b in zip(rank_sums,rr)]
        for v in set(vals):
            t=vals.count(v);ties+=t**3-t
    friedman=(12/(5*8*9)*sum(x*x for x in rank_sums)-3*5*9)/(1-ties/(5*(8**3-8)))
    z=friedman/2;q=math.erfc(math.sqrt(z));shape=0.5
    while shape<3.5:
        q+=z**shape*math.exp(-z)/math.gamma(shape+1);shape+=1
    for name,rows in [('per_seed_metrics.csv',runs),('model_summary.csv',aggregate),('per_seed_class_metrics.csv',perclass),('class_metrics_mean5.csv',classmean),('per_seed_confusion.csv',cells),('confusion_mean5.csv',mean_cells),('paired_seed_contrasts.csv',paired)]:write_csv(name,rows)
    (ROOT/'results/statistics'/'figure6_data.json').write_text(json.dumps(plot,indent=2),encoding='utf-8')
    report={'n_files':40,'n_test_per_run':570,'support':dict(zip(CLASSES,SUPPORT)),'S8':next(x for x in aggregate if x['model']=='S8'),'S8_seed_results':[x for x in runs if x['model']=='S8'],'friedman':{'statistic':friedman,'df':7,'asymptotic_p':q},'paired':paired,'note':'Intervals describe five-run variability conditional on the recorded test set; 2,850 predictions are not 2,850 independent physical kernels. sample_index is an evaluation row index, not a verified acquisition identifier.'}
    (ROOT/'results/statistics'/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
