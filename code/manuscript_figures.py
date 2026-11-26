#!/usr/bin/env python3
"""Regenerate four statistical figures from curated archived aggregates.
Optional PR from saved calibrated scores, never model inference or fitting.
The historical test population was previously exposed, not independent.
"""
import argparse, json, hashlib, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from retrospective_tables import tables, ORDER, HORIZONS

SERIES=[('primary_soft_full','Soft reference'),('full_hard','Full hard-label'),('observed_only_hard','Observed-only XGBoost'),('standalone_transformer','Standalone Transformer')]
FEATURE_LABELS={'momentum_if_event_end':'Event-flow momentum','observed_transport_60m_event_end':'Local transport count (60 min)','ewma_if_event_end':'Smoothed event flow','burstiness_cv_60m_event_end':'Event-count CV (60 min)','observed_transport_count_case_at_tau':'Case transport count','observed_transport_global_5m':'Global transport count (5 min)'}

def require(test,msg):
    if not test: raise ValueError(msg)

def setup():
    plt.style.use('default')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'pdf.fonttype':42,'axes.spines.top':True,'axes.spines.right':True,'axes.labelsize':13,'axes.titlesize':14,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':11,'legend.frameon':True,'legend.framealpha':.8,'legend.fancybox':True,'axes.grid':False,'axes.facecolor':'white','figure.facecolor':'white','savefig.facecolor':'white','lines.linewidth':1.5})

def save(f,out,name):
    f.savefig(out/name,bbox_inches='tight',pad_inches=.06);plt.close(f)

def validate(a):
    c=a['shap']['cohorts'];require([x['cohort'] for x in c]==['full_recorded_clock','strict_pre_archived_deadline','at_least_D5_before_archived_deadline','at_least_D10_before_archived_deadline'],'SHAP cohort order')
    require([(x['n_rows'],x['n_cases']) for x in c]==[(8908,1313),(6202,1309),(5566,1251),(4831,1160)],'SHAP population')
    for x in c:
        require(set(x['shares'])=={'observed19','teacher_logit','teacher_PCA48'} and all(0<=v<=1 for v in x['shares'].values()) and math.isclose(sum(x['shares'].values()),1,abs_tol=1e-10),'SHAP share sum')
    require([x['feature'] for x in c[2]['top_operational_features']]==list(FEATURE_LABELS),'SHAP feature order')
    require(all(0<=x['case_equal_abs_SHAP_share']<=1 for x in c[2]['top_operational_features']),'SHAP bounds')
    t=a['teacher'];h=t['history'];require(t['fit_queries']==67471 and t['early_stop_queries']==12313 and [x['epoch'] for x in h]==list(range(1,9)) and t['selected_epoch']==min(h,key=lambda x:x['stop_bce'])['epoch']==5,'Teacher history selection')
    require(all(math.isfinite(v) and v>=0 for x in h for v in x.values()),'Teacher metrics')
    require(all(x[k]<=1 for x in h for k in ['stop_AP','stop_weighted_AP','stop_AUROC']),'Teacher metrics bounds')

def aggregate_figures(d,a,out):
    tables(d);validate(a);setup()
    blue,orange,green,red,purple=plt.rcParams['axes.prop_cycle'].by_key()['color'][:5]; gray='0.45'
    cs=a['shap']['cohorts'];labels=['Full clock','Predeadline','≥5 min lead','≥10 min lead'];f,(ax,bx)=plt.subplots(1,2,figsize=(12,4.3),gridspec_kw={'width_ratios':[1.1,1.25]});left=np.zeros(4)
    for k,n,c in zip(['observed19','teacher_logit','teacher_PCA48'],['Operational (19)','Teacher logit','Teacher PCA48'],[blue,orange,green]):
        vals=np.array([100*x['shares'][k] for x in cs]);ax.barh(labels,vals,left=left,color=c,height=.56,label=n);left+=vals
    ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Share of absolute SHAP (%)');ax.set_axisbelow(True)
    v=cs[2]['top_operational_features'];bx.barh([FEATURE_LABELS[x['feature']] for x in v],[100*x['case_equal_abs_SHAP_share'] for x in v],color=blue,height=.55);bx.invert_yaxis();bx.set_xlabel('Share of all 68 inputs (%)');bx.set_title('Leading operational inputs (≥5 min)',fontsize=14);bx.set_axisbelow(True)
    for aa in (ax,bx):aa.tick_params(labelsize=13);aa.xaxis.label.set_size(14)
    ax.legend(fontsize=12,loc='lower center',bbox_to_anchor=(.5,1.01),frameon=True);f.tight_layout(w_pad=1.8);save(f,out,'v4427_shap.pdf')
    f,ax=plt.subplots(figsize=(8.71,4.75));h=list(map(int,HORIZONS))
    for (k,label),col in zip(SERIES,[blue,orange,green,red]):ax.plot(h,[d['models'][k]['horizons']['D5'][str(i)]['f1'] for i in h],label=label,color=col,lw=1.5)
    ax.set(xticks=h,ylim=(0,.72),xlabel='Minimum lead before archived deadline (min)',ylabel='Strict first-alarm F1');ax.set_title('First-Alarm Performance by Lead Time');ax.legend(loc='upper right');f.tight_layout();save(f,out,'v4427_horizons.pdf')
    t=a['teacher'];hist=t['history'];ep=[x['epoch'] for x in hist];best=t['selected_epoch'];f,(ax,bx)=plt.subplots(1,2,figsize=(12,4.65))
    for k,label,c in [('fit_bce','Weight fitting',blue),('stop_bce','Early stopping',orange)]:ax.plot(ep,[x[k] for x in hist],color=c,label=label)
    ax.set(xlabel='Epoch',ylabel='Interval-weighted BCE',xticks=ep);ax.set_title('Teacher Training Loss')
    for k,label,c in [('stop_AP','Unweighted AP',blue),('stop_weighted_AP','Interval-weighted AP',green),('stop_AUROC','Unweighted AUROC',orange)]:bx.plot(ep,[x[k] for x in hist],color=c,label=label)
    bx.set(xlabel='Epoch',ylabel='Early-stop metric',xticks=ep,ylim=(.55,1));bx.set_title('Teacher Performance Metrics')
    for aa in (ax,bx):
        aa.axvline(best,color=gray,ls='--',lw=1);aa.text(best+.09,aa.get_ylim()[1]-.035*(aa.get_ylim()[1]-aa.get_ylim()[0]),'Selected',color=gray,va='top',fontsize=10);aa.tick_params(labelsize=14);aa.xaxis.label.set_size(16);aa.yaxis.label.set_size(16);aa.title.set_size(16);aa.legend(loc='lower right',fontsize=13)
    ax.legend(loc='best',fontsize=13);f.tight_layout(w_pad=2);save(f,out,'v4427_teacher_training.pdf')
    f,ax=plt.subplots(figsize=(8.71,4.75));ax.plot([0,1],[0,1],ls='--',c='0.45',lw=1,label='Ideal reliability')
    for arm,label in [('primary_soft_full','Soft reference'),('full_hard','Full hard'),('observed_only_hard','Observed-only XGBoost')]:
        bins=[x for x in d['models'][arm]['calibration']['bins'] if x['count']];ax.plot([x['mean_probability'] for x in bins],[x['observed_positive_rate'] for x in bins],label=label)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed positive fraction',title='Probability Calibration');ax.legend(loc='upper left');f.tight_layout();save(f,out,'v4417_retrospective_reliability.pdf')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def pr_curves(d,frame):
    from sklearn.metrics import average_precision_score, precision_recall_curve
    require(len(frame)==d['cohort']['queries'] and frame['case_seg'].nunique()==d['cohort']['cases'] and not frame[['case_seg','k_idx','tau_ts']].isna().any().any() and not frame.duplicated(['case_seg','k_idx','tau_ts']).any(),'Score cohort or key mismatch')
    y=frame.next_static_delay.to_numpy();require(np.isin(y,[0,1]).all() and sum(y)==d['cohort']['positive_queries'],'Label mismatch');res={};curves={}
    for arm in ORDER:
        p=frame['prob_'+arm].to_numpy(dtype=float);require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid score')
        ap=float(average_precision_score(y,p));require(abs(ap-d['models'][arm]['primary']['average_precision'])<=1e-10,'Archived AP mismatch: '+arm);res[arm]=ap
        if arm in dict(SERIES):
            precision,recall,_=precision_recall_curve(y,p);curves[arm]=(recall,precision)
    return res,curves

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--retrospective-aggregates',type=Path,required=True);p.add_argument('--figure-aggregates',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--saved-scores',type=Path);p.add_argument('--scores-sha256');a=p.parse_args()
    require(bool(a.saved_scores)==bool(a.scores_sha256),'Supply saved scores and expected SHA-256 together');d=json.loads(a.retrospective_aggregates.read_text());f=json.loads(a.figure_aggregates.read_text());tables(d);validate(f);a.out_dir.mkdir(parents=True,exist_ok=False)
    aggregate_figures(d,f,a.out_dir);summary={'status':'FOUR_AGGREGATE_FIGURES_REGENERATED','new_training':False,'new_inference':False,'raw_test_event_read':False,'threshold_selection':False,'SHAP_recomputed':False,'teacher_history_retrained':False,'PR_status':'NOT_REQUESTED','independent_holdout':False}
    if a.saved_scores:
        import pandas as pd
        require(digest(a.saved_scores)==a.scores_sha256,'Saved score fingerprint mismatch')
        frame=pd.read_parquet(a.saved_scores,columns=['case_seg','k_idx','tau_ts','next_static_delay']+['prob_'+x for x in ORDER]);aps,curves=pr_curves(d,frame)
        require(digest(a.saved_scores)==a.scores_sha256,'Saved score file changed');fig,ax=plt.subplots(figsize=(8.71,4.75))
        for arm,label in SERIES:
            r,pr=curves[arm];ax.plot(r,pr,label=f'{label} (AP={aps[arm]:.4f})')
        ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='Recall',ylabel='Precision',title='Precision–Recall Curves');ax.legend(loc='best');fig.tight_layout();save(fig,a.out_dir,'v4430_retrospective_PR.pdf');summary.update(status='FOUR_AGGREGATE_FIGURES_AND_SAVED_SCORE_PR_PASS',PR_status='ALL_NINE_AP_MATCH',AP=aps)
    summary['figures']={p.name:digest(p) for p in sorted(a.out_dir.glob('*.pdf'))};(a.out_dir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
