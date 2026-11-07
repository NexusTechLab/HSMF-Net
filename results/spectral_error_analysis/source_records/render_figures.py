"""Render audited numerical exports; retain the original eight illustrative photographs."""
from pathlib import Path
import csv,json,hashlib,shutil
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parent
OLD=Path('F:/00 Paper-Summary/1.HSMF-Net-Paper/01-Experiments/MobileNet_Ablation_Code-V3/artifacts/sci_s1s6_tvt701515_2026-07-11_09-37/comparison/extra_data/error_analysis/sci_s6_fusion_hybrid_midfilm_spa24_seed42')
PAIRS=[('ZM22','XK129'),('CX186','ZM379')]
COLORS={'CX186':'#1f4e79','ZM379':'#9a6b16','ZM22':'#517653','XK129':'#a74f78'}
def read(name):return list(csv.DictReader((OUT/name).open(encoding='utf-8-sig')))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    report=json.loads((OUT/'analysis_report.json').read_text(encoding='utf-8'))
    spectra=read('training_class_mean_sd.csv');bands=read('training_bandwise_differences.csv')
    distance=report['new_training_results'];groups={c:[v for v in spectra if v['cultivar']==c] for c in COLORS}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'svg.fonttype':'none'})
    ymax=max(float(v['pointwise_bootstrap_95_high']) for v in bands)*1.34
    def plot_pair(top,bottom,col,letters):
        a,b=PAIRS[col]
        for c,ls in [(a,'-'),(b,'--')]:
            g=groups[c];w=np.array([float(v['wavelength_nm']) for v in g]);m=np.array([float(v['mean_processed_relative_reflectance']) for v in g]);sd=np.array([float(v['within_class_sample_sd']) for v in g]);keep=(w>=967.95)&(w<=1673.67)
            top.plot(w[keep],m[keep],color=COLORS[c],ls=ls,lw=1.6,label=f"{c} (n={g[0]['n_kernels']})")
            top.fill_between(w[keep],(m-sd)[keep],(m+sd)[keep],color=COLORS[c],alpha=.15,lw=0)
        top.axvspan(1200,1400,color='#888888',alpha=.08)
        top.set(xlim=(967.95,1673.67),ylim=(.075,.395),title=f'({letters[0]}) Training means ± within-class SD',xlabel='Wavelength (nm)',ylabel='Processed relative reflectance')
        top.legend(loc='upper right',frameon=False)
        v=[r for r in bands if r['pair']==f'{a} vs {b}'];xx=[float(r['wavelength_nm']) for r in v]
        bottom.plot(xx,[float(r['absolute_class_mean_difference']) for r in v],color='#294a60',lw=1.7)
        bottom.fill_between(xx,[float(r['pointwise_bootstrap_95_low']) for r in v],[float(r['pointwise_bootstrap_95_high']) for r in v],color='#7699b1',alpha=.3,lw=0)
        bottom.set(xlim=(1201.28,1399.09),ylim=(0,ymax),title=f'({letters[1]}) {a} vs {b}',xlabel='Wavelength (nm)',ylabel='Absolute class-mean difference')
        d=distance[col]
        bottom.text(.025,.97,f"D = {d['mean_absolute_class_mean_difference']:.5f}\n95% interval [{d['bootstrap_percentile_95_low']:.5f}, {d['bootstrap_percentile_95_high']:.5f}]",transform=bottom.transAxes,ha='left',va='top',fontsize=8.5)
        for ax in [top,bottom]:
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#dddddd',lw=.6,alpha=.7)
    fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for col in range(2):plot_pair(ax[0,col],ax[1,col],col,['ab'[col],'cd'[col]])
    fig.savefig(OUT/'figure_S20_training_spectra.png',dpi=300);fig.savefig(OUT/'figure_S20_training_spectra.svg');plt.close(fig)

    case_source=list(csv.DictReader((OLD/'misclassified_samples_index.csv').open(encoding='utf-8-sig')))
    case_manifest=[];case_groups=[]
    photo_dir=OUT/'illustrative_cases';photo_dir.mkdir(exist_ok=True)
    for a,b in PAIRS:
        chosen=[v for v in case_source if v['true_class']==a and v['pred_class']==b][:4]
        assert len(chosen)==4
        group=[]
        for v in chosen:
            src=OLD/v['camera_rgb']
            resolution='archived portable camera crop'
            if not src.exists():
                src=OLD.parents[5]/'data/wheat-seed'/a/'rgb'/(v['stem']+'.png')
                resolution='current same-stem camera crop resolved from archived sample identifier'
            dest=photo_dir/(v['sample_id']+'.png')
            shutil.copy2(src,dest)
            item={'sample_id':v['sample_id'],'true_class':a,'pred_class':b,'archived_confidence':float(v['confidence']),
                  'source_run':'sci_s6_fusion_hybrid_midfilm_spa24_seed42','source_image':str(src),
                  'portable_image':str(dest.relative_to(OUT)),'image_sha256':sha(dest),
                  'image_resolution':resolution,'status':'retained archived sample ID and confidence; no new model prediction'}
            case_manifest.append(item);group.append(item)
        case_groups.append(group)
    (OUT/'illustrative_case_manifest.json').write_text(json.dumps(case_manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fig=plt.figure(figsize=(13,8))
    grid=fig.add_gridspec(2,3,width_ratios=[.95,1.35,1.35],hspace=.38,wspace=.26,left=.025,right=.99,bottom=.065,top=.95)
    photos=grid[:,0].subgridspec(4,2,hspace=.07,wspace=.06)
    for col,group in enumerate(case_groups):
        for row,v in enumerate(group):
            ax=fig.add_subplot(photos[row,col]);ax.imshow(Image.open(OUT/v['portable_image']));ax.set_axis_off()
            ax.text(.035,.96,f"{100*v['archived_confidence']:.0f}%",transform=ax.transAxes,ha='left',va='top',color='white',fontsize=8,bbox={'facecolor':'black','edgecolor':'none','pad':1,'alpha':.7})
            if row==0:
                a,b=PAIRS[col];ax.set_title(f'({"ab"[col]}) {a} → {b}',fontsize=9,pad=5)
    for col in range(2):plot_pair(fig.add_subplot(grid[0,col+1]),fig.add_subplot(grid[1,col+1]),col,['cd'[col],'ef'[col]])
    fig.savefig(OUT/'figure_11_training_revised.png',dpi=300);fig.savefig(OUT/'figure_11_training_revised.svg');plt.close(fig)
    report['rendering']={'source':str(Path(__file__).relative_to(OUT)),'source_sha256':sha(Path(__file__)),
                         'common_pairwise_y_scales':True,'main_figure':'figure_11_training_revised.png',
                         'case_manifest_sha256':sha(OUT/'illustrative_case_manifest.json'),
                         'notes':'Original eight archived images/confidences retained. Statistical panels use current TRAIN spectra, no individual error-spectrum overlays. SD upper / pointwise bootstrap lower; scalar D intervals in annotations. Original all-collection distances traced separately.'}
    (OUT/'analysis_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Rendered revised Figure 11 and Figure S20 with common scales; retained original eight photographs and source confidence records.')

if __name__=='__main__':main()
