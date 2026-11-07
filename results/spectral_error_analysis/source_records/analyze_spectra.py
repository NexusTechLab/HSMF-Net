"""Audit actual training HSI spectra and bootstrap two prespecified descriptive distances.

No model inference/training; validation and test cube contents are not read.
"""
from pathlib import Path
from collections import Counter
from typing import Any
import ast,csv,hashlib,json,sys,platform,re
import numpy as np
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
SUPP=OUT.parent
FIT=SUPP/'spectral_metadata/controlled_refit_2026-11-05'
PROJECT=Path('F:/00 Paper-Summary/1.HSMF-Net-Paper/01-Experiments/MobileNet_Ablation_Code-V3')
OLD=PROJECT/'artifacts/sci_s1s6_tvt701515_2026-07-11_09-37/comparison/extra_data/error_analysis/sci_s6_fusion_hybrid_midfilm_spa24_seed42'
CLASSES=['CX186','XK129','ZM22','ZM379']
PAIRS=[('ZM22','XK129'),('CX186','ZM379')]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save_json(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def save_csv(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ledger=json.loads((FIT/'training_content_manifest.json').read_text(encoding='utf-8'))
    fingerprint=json.loads((FIT/'input_fingerprint.json').read_text(encoding='utf-8'))
    records=ledger['records']
    assert len(records)==2660 and all(x['split']=='train' for x in records)
    assert len({x['sample_id'] for x in records})==2660
    assert Counter(x['cultivar'] for x in records)=={'CX186':630,'XK129':700,'ZM22':700,'ZM379':630}
    assert sha(FIT/'training_spectra_preprocessed.npy')==ledger['matrix_sha256']
    source=PROJECT/'src/mobilenet_enterprise/data/wheat_seed_hsi.py'
    assert sha(source)==fingerprint['source_sha256'],'Source changed relative to refit'
    names={'_detect_anomalous_bands','_smooth_boundary_noise','_preprocess_spectrum_for_spa'}
    nodes=[n for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==3
    ns={'np':np,'Any':Any}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    raw=np.load(FIT/'training_spectra_raw.npy',allow_pickle=False)
    processed=np.load(FIT/'training_spectra_preprocessed.npy',allow_pickle=False)
    wl=np.load(FIT/'training_wavelengths.npy',allow_pickle=False).astype('float64')
    assert raw.shape==processed.shape==(2660,454) and np.isfinite(raw).all() and np.isfinite(processed).all()
    region=(wl>=1200)&(wl<=1400)
    ids=np.flatnonzero(region)
    assert len(ids)==121 and (ids[0],ids[-1])==(152,272)
    kernel_rows=[];max_raw_error=0.;max_processed_error=0.;cube_bytes=0
    for i,record in enumerate(records):
        hdr=Path(record['hdr']['path']);rawpath=Path(record['hsi_raw']['path'])
        assert sha(hdr)==record['hdr']['sha256'] and sha(rawpath)==record['hsi_raw']['sha256']
        text=hdr.read_text(encoding='utf-8',errors='replace')
        def scalar(key):return int(re.search(r'^'+re.escape(key)+r'\s*=\s*(\d+)',text,re.I|re.M).group(1))
        h,w,b=scalar('lines'),scalar('samples'),scalar('bands')
        assert b==454 and scalar('data type')==4 and scalar('header offset')==0
        interleave=re.search(r'^interleave\s*=\s*(\w+)',text,re.I|re.M).group(1).lower()
        values=np.fromstring(re.search(r'wavelength\s*=\s*\{([^}]+)\}',text,re.I|re.S).group(1),sep=',')
        assert np.array_equal(values,wl)
        dtype=np.dtype('<f4' if scalar('byte order')==0 else '>f4')
        data=np.fromfile(rawpath,dtype=dtype)
        assert data.size==h*w*b
        if interleave=='bil':cube=data.reshape(h,b,w).transpose(0,2,1)
        elif interleave=='bsq':cube=data.reshape(b,h,w).transpose(1,2,0)
        elif interleave=='bip':cube=data.reshape(h,w,b)
        else:raise ValueError(interleave)
        assert np.isfinite(cube).all()
        flat=cube.reshape(h*w,b);mask=flat.sum(axis=1)>1e-6
        n=int(mask.sum());assert n>0,'No-foreground fallback is not permitted in this analysis'
        mean=flat[mask].mean(axis=0).astype('float32')
        repaired=ns['_preprocess_spectrum_for_spa'](mean)
        e=float(np.max(np.abs(mean-raw[i])));ep=float(np.max(np.abs(repaired-processed[i])))
        max_raw_error=max(max_raw_error,e);max_processed_error=max(max_processed_error,ep)
        assert e<=1e-7 and ep<=1e-7
        kernel_rows.append({'sample_id':record['sample_id'],'cultivar':record['cultivar'],'split':'train',
                            'foreground_pixels':n,'crop_pixels':h*w,'hdr_sha256':record['hdr']['sha256'],
                            'hsi_sha256':record['hsi_raw']['sha256'],
                            'mean_window_reflectance':float(processed[i,region].astype('float64').mean())})
        cube_bytes+=rawpath.stat().st_size
        if (i+1)%400==0 or i+1==len(records):print(f'Authenticated TRAIN spectra and foreground counts: {i+1}/2660',flush=True)
    save_csv('training_kernel_ledger.csv',kernel_rows)
    x=processed.astype('float64');unrepaired=raw.astype('float64');group={c:x[[v['cultivar']==c for v in records]] for c in CLASSES}
    pixel_rows=[];spectral_rows=[]
    for c in CLASSES:
        pix=np.array([v['foreground_pixels'] for v in kernel_rows if v['cultivar']==c])
        g=group[c];mean=g.mean(axis=0);sd=g.std(axis=0,ddof=1)
        pixel_rows.append({'cultivar':c,'n_kernels':len(g),'foreground_pixels_total':int(pix.sum()),
                           'foreground_pixels_min':int(pix.min()),'foreground_pixels_median':float(np.median(pix)),
                           'foreground_pixels_mean':float(pix.mean()),'foreground_pixels_max':int(pix.max()),
                           'mean_within_class_sd_1200_1400':float(sd[region].mean())})
        for j in range(454):spectral_rows.append({'cultivar':c,'n_kernels':len(g),'band_index_zero_based':j,'wavelength_nm':wl[j],
                                                 'mean_processed_relative_reflectance':mean[j],'within_class_sample_sd':sd[j]})
    save_csv('class_kernel_pixel_counts.csv',pixel_rows);save_csv('training_class_mean_sd.csv',spectral_rows)
    n_boot=2000;rng=np.random.default_rng(20261105);boot_means={}
    with threadpool_limits(limits=1):
        for c in CLASSES:
            g=group[c][:,region];n=len(g)
            counts=rng.multinomial(n,np.full(n,1/n),size=n_boot)
            boot_means[c]=counts@g/n
    result=[];band_rows=[];draw_rows=[]
    for a,b in PAIRS:
        diff=np.abs(group[a].mean(0)[region]-group[b].mean(0)[region]);d=float(diff.mean())
        bootdiff=np.abs(boot_means[a]-boot_means[b]);draws=bootdiff.mean(axis=1)
        lo,hi=np.quantile(draws,[.025,.975],method='linear')
        result.append({'cultivar_a':a,'cultivar_b':b,'split':'train','n_a':len(group[a]),'n_b':len(group[b]),
                       'n_wavelengths':121,'wavelength_low_nm':float(wl[region][0]),'wavelength_high_nm':float(wl[region][-1]),
                       'mean_absolute_class_mean_difference':d,'bootstrap_percentile_95_low':float(lo),
                       'bootstrap_percentile_95_high':float(hi),'bootstrap_repetitions':n_boot})
        lower,upper=np.quantile(bootdiff,[.025,.975],axis=0,method='linear')
        for k,j in enumerate(ids):band_rows.append({'pair':f'{a} vs {b}','band_index_zero_based':int(j),'wavelength_nm':wl[j],
                                                   'absolute_class_mean_difference':diff[k],'pointwise_bootstrap_95_low':lower[k],
                                                   'pointwise_bootstrap_95_high':upper[k]})
        for k,v in enumerate(draws):draw_rows.append({'pair':f'{a} vs {b}','bootstrap_draw':k+1,'mean_absolute_difference':float(v)})
    save_csv('training_pair_distances.csv',result);save_csv('training_bandwise_differences.csv',band_rows);save_csv('bootstrap_distance_draws.csv',draw_rows)
    old_rows=list(csv.DictReader((OLD/'class_mean_std_spectra.csv').open(encoding='utf-8-sig')))
    old_check=[]
    for a,b in PAIRS:
        ra=[v for v in old_rows if v['class_name']==a];rb=[v for v in old_rows if v['class_name']==b]
        assert len(ra)==len(rb)==454
        wa=np.array([float(v['wavelength_nm']) for v in ra]);wb=np.array([float(v['wavelength_nm']) for v in rb])
        assert np.array_equal(wa,wl) and np.array_equal(wb,wl)
        da=np.array([float(v['mean_reflectance']) for v in ra]);db=np.array([float(v['mean_reflectance']) for v in rb])
        d=float(np.abs(da[region]-db[region]).mean())
        old_check.append({'cultivar_a':a,'cultivar_b':b,'n_a_recorded':int(ra[0]['n_samples']),'n_b_recorded':int(rb[0]['n_samples']),
                          'archived_mean_distance_recomputed':d,'rounded_4_decimals':f'{d:.4f}',
                          'scope':'archived full-collection class-mean summary; not train-only',
                          'source_sha256':sha(OLD/'class_mean_std_spectra.csv')})
    save_csv('historical_point_estimate_reproduction.csv',old_check)
    hashes={str(p.relative_to(SUPP)) if p.is_relative_to(SUPP) else str(p):sha(p) for p in [Path(__file__),FIT/'training_content_manifest.json',FIT/'training_spectra_raw.npy',FIT/'training_spectra_preprocessed.npy',FIT/'training_wavelengths.npy',source,OLD/'class_mean_std_spectra.csv',PROJECT/'scripts/export_error_analysis_samples.py']}
    report={'status':'COMPLETED_REAL_TRAINING_SPECTRAL_ANALYSIS','date':'2026-11-05','n_kernels':2660,'unique_filename_ids':2660,
            'scope':'Current frozen training manifest only; no validation/test cube contents or predictions read. Historical summary read separately for point-estimate tracing.',
            'representation':'Foreground-mean exported relative-reflectance spectra after per-kernel SPA-style anomaly repair/boundary treatment; before training-column z-scoring and network normalization.',
            'sg':'Author-confirmed export workflow: pixel-level SG window11/order3/derivative0/nearest. No extra SG applied here.',
            'foreground_definition':'sum of 454 band values > 1e-6; each kernel has equal weight in its class mean; no pixel-weighted pooling across kernels.',
            'exclusions':'No included training records removed; finite spectra and nonempty foreground verified. No new kernel outlier exclusion or winsorization.',
            'wavelength_window':{'band_indices_zero_based':[152,272],'n_bands':121,'actual_endpoints_nm':[float(wl[region][0]),float(wl[region][-1])],'equal_band_weighting':True,'spa24_subset_only':False},
            'validation':{'all_training_hdr_raw_hashes_match':True,'max_cached_raw_recompute_abs_error':max_raw_error,'max_cached_processed_recompute_abs_error':max_processed_error,
                          'raw_vs_processed_max_difference_in_window':float(np.abs(x[:,region]-unrepaired[:,region]).max()),'hsi_bytes_read':cube_bytes,
                          'processed_window_range':[float(x[:,region].min()),float(x[:,region].max())]},
            'bootstrap':{'n':n_boot,'rng':'NumPy Generator PCG64','seed':20261105,'resampling':'independent within-cultivar multinomial counts equivalent to sampling n kernels with replacement','interval':'95% percentile, quantile method linear',
                         'limits':'Conditional kernel/file-level uncertainty for this collection, not lot/session-resampled or external-source uncertainty. Bandwise intervals are pointwise, not simultaneous. Nonnegative absolute-distance intervals are not a null-hypothesis separation test.'},
            'class_counts':pixel_rows,'new_training_results':result,'historical_reproduction':old_check,
            'historical_trace_limit':'Archived means and exporter document full-collection equal-kernel summaries; the original per-kernel foreground pixel ledger was not retained. New training analysis supplies an authenticated current ledger.',
            'chemical_reference':'No biochemical measurements inferred; no retraining/model selection performed.',
            'analysis_timing':'Post hoc descriptive analysis responding to the reviewer; no new tuning of bands, normalization or model hyperparameters.',
            'environment':{'python':platform.python_version(),'numpy':np.__version__},'source_hashes':hashes}
    save_json('analysis_report.json',report)
    # Standalone manuscript-compatible scientific figure; SD and CI meanings remain distinct.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':8,'svg.fonttype':'none'})
    colors={'CX186':'#1f4e79','ZM379':'#9a6b16','ZM22':'#517653','XK129':'#a74f78'}
    fig,axes=plt.subplots(2,2,figsize=(11,7.3),layout='constrained')
    selected=(wl>=967.95)&(wl<=1673.67)
    for col,(a,b) in enumerate(PAIRS):
        ax=axes[0,col]
        for c,ls in [(a,'-'),(b,'--')]:
            m=group[c].mean(0);sd=group[c].std(0,ddof=1)
            ax.plot(wl[selected],m[selected],color=colors[c],ls=ls,lw=1.6,label=f'{c} (n={len(group[c])})')
            ax.fill_between(wl[selected],(m-sd)[selected],(m+sd)[selected],color=colors[c],alpha=.14,lw=0)
        ax.axvspan(1200,1400,color='#888888',alpha=.09)
        ax.set(title=f'({chr(97+col)}) Training means ± within-class SD',xlabel='Wavelength (nm)',ylabel='Processed relative reflectance')
        ax.legend(loc='upper right');ax.set_xlim(967.95,1673.67)
        ax=axes[1,col];pair=result[col]
        bands=[v for v in band_rows if v['pair']==f'{a} vs {b}'];xx=[v['wavelength_nm'] for v in bands]
        ax.plot(xx,[v['absolute_class_mean_difference'] for v in bands],color='#294a60',lw=1.7,label='Observed mean-spectrum difference')
        ax.fill_between(xx,[v['pointwise_bootstrap_95_low'] for v in bands],[v['pointwise_bootstrap_95_high'] for v in bands],color='#7699b1',alpha=.3,lw=0,label='Pointwise 95% bootstrap interval')
        ax.set(title=f'({chr(99+col)}) {a} vs {b}',xlabel='Wavelength (nm)',ylabel='Absolute class-mean difference')
        ax.set_xlim(wl[region][0],wl[region][-1]);ax.set_ylim(bottom=0)
        ax.text(.02,.96,f"D = {pair['mean_absolute_class_mean_difference']:.5f}\n95% interval [{pair['bootstrap_percentile_95_low']:.5f}, {pair['bootstrap_percentile_95_high']:.5f}]",transform=ax.transAxes,ha='left',va='top',fontsize=9)
        ax.legend(loc='lower left',fontsize=7)
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#dddddd',lw=.6,alpha=.7)
    fig.savefig(OUT/'figure_S20_training_spectra.png',dpi=300)
    fig.savefig(OUT/'figure_S20_training_spectra.svg')
    plt.close(fig)
    print(json.dumps({'historical':old_check,'training':result,'pixel_counts':pixel_rows},ensure_ascii=True,indent=2),flush=True)

if __name__=='__main__':main()
