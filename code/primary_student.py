#!/usr/bin/env python3
"""Portable primary teacher–student recipe for the processed research inputs.

Prepare uses an existing label-free teacher export and frozen PCA; fit trains
only the primary student and weighted Platt calibrator from pinned matrices.
Neither operation trains the teacher/PCA, selects thresholds, accesses held-out
splits, or replaces the reported frozen model. Exact archived refit is not
claimed; the original-source control reproduced the same refit discrepancy.
Run only with authorized input artifacts. joblib inputs must be trusted and
match their recorded hashes. Output directories must not already exist.
"""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
VERSION='1.0'
KEY=['case_seg','k_idx','tau_ts']
BASE=[f'{v}_{w}m_event_end' for w in (5,15,60) for v in ('starts','burstiness_cv')]+['new_cases_1h','active_cases_1h','ewma_if_event_end','momentum_if_event_end']+[f'observed_transport_{w}m_event_end' for w in (5,15,60)]+[f'observed_transport_global_{w}m' for w in (5,15,60)]+['observed_transport_count_case_at_tau','seconds_since_last_observed_transport','seconds_since_observed_transport_t0']
EMBED=[f'teacher_embed_{i:03d}' for i in range(192)]
PC=[f'teacher_pca_{i:02d}' for i in range(48)]
COLS=BASE+['teacher_logit']+PC
LABELS=['y_next_delay','weight_predeadline','is_predeadline']
META=['t0','t_next','next_static_plan']
DATA_PINS={'train_query_index':'81a78054d1980de1c1bd888d459a7b773d87191a79412ce140c097a503dd7cbe','train_student_candidates':'c99d0eadc5be09f11a948824055989894e00b068e6d1a694ce91196c340e3bc5','train_query_labels':'a1541b655c5831ba2f513efbe28eb7e04a205802b2ab08c6195578deee72c491','train_retrospective_metadata':'0c09a13afcac4e3dab1276709fd0e05606476dc973fc87f8591c4541c50a5e62'}
OOS_PIN='c7bb0f9110c084f92c0118df59ec1335687c64963c7c241c6f486ea6c9b1c3a8'
PCA_PIN='9ff3edd01c3ff38477cb967f606f9aff9dbdb3a3057d082c642861bf2cc08ac5'
PERIOD_COUNTS={'P0':(32493,3108),'P1':(6253,996),'P2':(8918,1093),'P3':(6767,1267),'P4':(6595,1032),'P5':(8908,1313),'PURGED':(12852,1502)}
STAGES={'fit':(['P0','P1','P2','P3'],True,31080,6431,7927),'calibration':(['P4'],True,5163,1026,1683),'threshold_selection':(['P5'],False,8908,1313,4367)}
import tempfile
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
CFG=dict(max_depth=5,min_child_weight=5,learning_rate=.05,subsample=.8,colsample_bytree=.9,reg_lambda=2.,reg_alpha=0.,objective='binary:logistic',tree_method='hist',eval_metric='logloss',seed=42,nthread=4)
N_TREES=400
THRESHOLDS=(.24,.35)
TOL=1e-6
MATRIX_PINS={
 'fit':{'predictors':'2b4d0217765fb4340b83be0b4c4cda10e22bf1852bc57abfc10a040975be3313','supervision':'a6f49cb5bbf4694df5d2782b9437592ec023c9e6c75e23fb13c5726046e53847','retrospective_metadata':'5f13bf2610f88882f022c5e1aa4382d7f91d16b92b309fef57e112839b474b89'},
 'calibration':{'predictors':'407c9ec9cd009716d8f61c16db5527116cafd166012c653ff68a6dcbe1fec585','supervision':'18e66008ba94da4944cb4828dc91d46dfb0ec0f3afa0303dcaa7d5a3848c5a39','retrospective_metadata':'bf7dfed29e2af7f7a6be448fb4dac5612d63a27eb6cdff46e78921e36b7533ad'},
 'threshold_selection':{'predictors':'dc71d995e6781db24e05b53ebb94a981361a013b1542056a1d47d43230157c1c','supervision':'5a3a2dae1ba28716600e6bd5d0bbb47c7984257b460dc0b764cf659940f9e89e','retrospective_metadata':'2ce3538994cf82c02dfc5a86e5418e45a8dd4fbfbfdae7a6cb13ded98859fba4'}}
COUNTS={'fit':(31080,6431,7927),'calibration':(5163,1026,1683),'threshold_selection':(8908,1313,4367)}
class GateError(Exception):pass
def need(x,code):
 if not bool(x):raise GateError(code)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def pin(p,h):
 need(p.is_file() and not p.is_symlink() and sha(p)==h,'INPUT_PIN_MISMATCH');return p
def utc(s):return pd.to_datetime(s,utc=True).astype('datetime64[ns, UTC]')
def keyed(d):
 d=d.copy();d['case_seg']=d.case_seg.astype('string').str.strip();d['k_idx']=pd.to_numeric(d.k_idx,errors='raise').astype('int64');d['tau_ts']=utc(d.tau_ts)
 need(d.columns.is_unique and d[KEY].notna().all().all() and d.case_seg.ne('').all() and not d.duplicated(KEY).any(),'BAD_KEYS');return d
def join(a,b):
 need(len(a)==len(b) and not (set(a.columns)&set(b.columns)-set(KEY)),'JOIN_SCHEMA')
 z=a.merge(b,on=KEY,how='left',sort=False,validate='one_to_one',indicator=True)
 need(z._merge.eq('both').all() and z[KEY].equals(a[KEY].reset_index(drop=True)),'JOIN_COVERAGE_OR_ORDER');return z.drop(columns='_merge')
def periods(d):
 g=d.groupby('case_seg',observed=True,sort=False).agg(first=('t0','min'),last=('t_next','max')).sort_values('first',kind='stable')
 need(len(g)>=48 and g.notna().all().all() and g['last'].ge(g['first']).all(),'INVALID_CASE_SPANS')
 cuts=[g.iloc[math.floor(len(g)*q)]['first'] for q in (.32,.45,.58,.72,.86)]
 need(all(a<b for a,b in zip(cuts,cuts[1:])),'TIED_CUTOFFS')
 a=pd.Series('PURGED',index=g.index,dtype='string');gap=pd.Timedelta(hours=2)
 for i in range(6):
  ok=pd.Series(True,index=g.index)
  if i:ok &= g['first']>cuts[i-1]+gap
  if i<5:ok &= g['last']<cuts[i]-gap
  need(a[ok].eq('PURGED').all(),'ROLE_OVERLAP');a.loc[ok]=f'P{i}'
 return a
def soft(y,logit):
 y=np.asarray(y,dtype=np.float64);logit=np.asarray(logit,dtype=np.float64)
 need(y.shape==logit.shape and y.ndim==1 and len(y)>0 and np.isin(y,[0.,1.]).all() and np.isfinite(logit).all(),'SOFT_TARGET_INPUT')
 z=.75*y+.25/(1.+np.exp(-np.clip(logit/2.,-60.,60.)))
 need(np.isfinite(z).all() and ((z>0)&(z<1)).all(),'SOFT_TARGET_RANGE');return z
def predictors(d,pca):
 x=d[BASE+['teacher_logit']].copy();z=pca.transform(d[EMBED].to_numpy(np.float32))
 need(z.shape==(len(d),48) and np.isfinite(z).all(),'PCA_SHAPE')
 x=pd.concat([x.reset_index(drop=True),pd.DataFrame(z.astype(np.float32),columns=PC)],axis=1)
 need(list(x)==COLS and np.isfinite(x.to_numpy(np.float32)).all(),'FEATURE_MATRIX');return x


def logodds(p):
 p=np.asarray(p,dtype=np.float64);need(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'BAD_PROBABILITY');p=np.clip(p,1e-6,1-1e-6);return np.log(p/(1-p)).reshape(-1,1)

def fit_student(x,y,w,n=N_TREES):
 return xgb.train(CFG,xgb.DMatrix(np.asarray(x,np.float32),label=np.asarray(y,np.float32),weight=np.asarray(w,np.float64)),num_boost_round=n)

def predict(model,x):return model.predict(xgb.DMatrix(np.asarray(x,np.float32)))

def calibrate(raw,y,w):
 need(set(np.unique(y))=={0,1} and np.isfinite(w).all() and (w>0).all(),'CALIBRATION_INPUT')
 c=LogisticRegression(C=1000,solver='lbfgs',max_iter=1000,random_state=42);c.fit(logodds(raw),np.asarray(y,np.int8),sample_weight=w);need(c.coef_[0,0]>0,'CALIBRATION_SLOPE');return c

def calibrated(c,p):return c.predict_proba(logodds(p))[:,1]

def compare(a,b):
 a=np.asarray(a,np.float64);b=np.asarray(b,np.float64);need(a.shape==b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all(),'COMPARE_INPUT');d=np.abs(a-b)
 return dict(rows=int(a.size),exact_rows=int((d==0).sum()),max_abs_difference=float(d.max()),mean_abs_difference=float(d.mean()),rows_above_tolerance=int((d>TOL).sum()))

def read_stage(root,name,paths):
 t={}
 for k,h in MATRIX_PINS[name].items():
  p=pin(root/f'{name}_{k}.parquet',h);paths[p]=h;t[k]=pd.read_parquet(p)
 x,s,m=(t[k] for k in ('predictors','supervision','retrospective_metadata'))
 need(list(x)==KEY+COLS and list(s)==KEY+LABELS+(['soft_target'] if name=='fit' else []) and list(m)==KEY+META,'SCHEMA')
 need(x[KEY].notna().all().all() and not x.duplicated(KEY).any() and x[KEY].equals(s[KEY]) and x[KEY].equals(m[KEY]),'KEY_ORDER')
 need((len(x),x.case_seg.nunique(),int(s.y_next_delay.sum()))==COUNTS[name],'CENSUS')
 need(np.isfinite(x[COLS].to_numpy(np.float32)).all() and s.y_next_delay.isin([0,1]).all(),'INPUT_RANGE')
 times={k:pd.to_datetime(m[k],utc=True) for k in META};tau=pd.to_datetime(x.tau_ts,utc=True)
 need(all(v.notna().all() for v in times.values()) and (tau>=times['t0']).all() and (tau<times['t_next']).all() and np.array_equal(tau<times['next_static_plan'],s.is_predeadline.astype(bool)),'CLOCK')
 if name!='threshold_selection':need(s.is_predeadline.astype(bool).all() and np.isfinite(s.weight_predeadline).all() and s.weight_predeadline.gt(0).all(),'PREDEADLINE')
 if name=='fit':need(np.array_equal(s.soft_target.to_numpy(),soft(s.y_next_delay,x.teacher_logit)),'SOFT_TARGET')
 return x,s,m

def matrix_selftest():
 t=pd.Timestamp('2025-01-01',tz='UTC');d=pd.DataFrame({'case_seg':[str(i) for i in range(120)],'t0':[t+pd.Timedelta(hours=10*i) for i in range(120)]});d['t_next']=d.t0+pd.Timedelta(minutes=5)
 a=periods(d);z=d.copy();z.t0+=pd.Timedelta(weeks=8);z.t_next+=pd.Timedelta(weeks=8);need(a.equals(periods(z)) and a.eq('PURGED').sum()==5,'SYNTHETIC_CHRONOLOGY')
 need(np.array_equal(soft([0,1],[0,0]),[.125,.875]),'SYNTHETIC_SOFT')
 keys=pd.DataFrame({'case_seg':['b','a'],'k_idx':[1,1],'tau_ts':[t,t]});keys=keyed(keys);other=keys.iloc[::-1].assign(value=[2,1]);need(join(keys,other).value.tolist()==[1,2],'SYNTHETIC_JOIN_ORDER')
 class DummyPCA:
  def transform(self,x):return x[:,:48]
 x=pd.DataFrame(np.zeros((2,212),np.float32),columns=BASE+['teacher_logit']+EMBED);need(predictors(x,DummyPCA()).shape==(2,68),'SYNTHETIC_FEATURE_ORDER')
 print('Synthetic chronology, key-order, soft-target and 68-feature checks passed.')
def training_selftest():
 rng=np.random.default_rng(42);x=rng.normal(size=(160,68)).astype(np.float32);y=(x[:,0]>0).astype(np.int8);w=np.ones(160);z=soft(y,x[:,0]);need(np.all((z>0)&(z<1)),'SYNTHETIC_SOFT')
 a=fit_student(x,z,w,8);p=predict(a,x);c=calibrate(p,y,w);q=calibrated(c,p)
 with tempfile.TemporaryDirectory() as td:
  path=Path(td)/'synthetic.json';a.save_model(str(path));b=xgb.Booster();b.load_model(str(path));need(compare(p,predict(b,x))['max_abs_difference']==0,'SYNTHETIC_SERIALIZATION')
 need(compare(q,q)['rows_above_tolerance']==0 and compare(q,q+1e-4)['rows_above_tolerance']==len(q),'SYNTHETIC_COMPARISON')
 print('Synthetic student fit, Platt calibration, serialization and comparison checks passed.')

def prepare(args,out,r):
 base=Path(args.data_dir)
 p={n:pin(base/(n+'.parquet'),h) for n,h in DATA_PINS.items()}
 oos=pin(Path(args.teacher_export),OOS_PIN)
 pca_path=pin(Path(args.pca),PCA_PIN)
 idx=keyed(pd.read_parquet(p['train_query_index'],columns=KEY+['part'],filters=[('part','=','student_fit')]))
 need(len(idx)==82786 and idx.case_seg.nunique()==10311 and idx.part.eq('student_fit').all(),'STUDENT_COHORT')
 f=idx[KEY].copy();flt=[('case_seg','in',idx.case_seg.unique().tolist())]
 for n,cols in [('train_student_candidates',BASE),('train_query_labels',LABELS),('train_retrospective_metadata',META)]:
  f=join(f,keyed(pd.read_parquet(p[n],columns=KEY+cols,filters=flt)))
 teacher=keyed(pd.read_parquet(oos,columns=KEY+['teacher_logit']+EMBED));need(f[KEY].equals(teacher[KEY]),'ARCHIVED_OOS_ORDER_CHANGED');f=join(f,teacher)
 for c in META:f[c]=utc(f[c])
 need(f[META].notna().all().all() and f.y_next_delay.isin([0,1]).all() and f.tau_ts.ge(f.t0).all() and f.tau_ts.lt(f.t_next).all() and np.array_equal(f.tau_ts.lt(f.next_static_plan),f.is_predeadline.astype(bool)),'CLOCK_LABEL')
 need(f.groupby(['case_seg','k_idx'],observed=True).y_next_delay.nunique().eq(1).all(),'INTERVAL_LABEL')
 f['period']=f.case_seg.map(periods(f));counts={k:(int(f.period.eq(k).sum()),int(f.loc[f.period.eq(k),'case_seg'].nunique())) for k in PERIOD_COUNTS};need(counts==PERIOD_COUNTS,'PERIOD_CENSUS')
 pca=joblib.load(pca_path);need(pca.n_features_in_==192 and pca.n_components_==48 and pca.n_samples_==31080 and pca.svd_solver=='randomized' and pca.random_state==42,'FROZEN_PCA_CONTRACT')
 # Original V4.337 transforms each stage separately; do not concatenate the PCA calls.
 result={};parts={}
 for name,(roles,early,n,ncase,pos) in STAGES.items():
  z=f[f.period.isin(roles)].copy();parts[name]=z
  if early:z=z[z.is_predeadline.astype(bool)].copy()
  z=z.reset_index(drop=True);need((len(z),z.case_seg.nunique(),int(z.y_next_delay.sum()))==(n,ncase,pos),'STAGE_CENSUS_'+name)
  if early:need(z.weight_predeadline.gt(0).all() and np.isfinite(z.weight_predeadline).all(),'WEIGHTS')
  x=predictors(z,pca);x=pd.concat([z[KEY],x],axis=1);s=z[KEY+LABELS].copy();m=z[KEY+META].copy()
  if name=='fit':s['soft_target']=soft(s.y_next_delay,z.teacher_logit)
  # Separate predictors from labels and retrospective timing; no timestamps/labels in feature allowlist.
  dest=out/'matrices';dest.mkdir(exist_ok=True)
  hashes={}
  for suffix,table in [('predictors',x),('supervision',s),('retrospective_metadata',m)]:
   path=dest/f'{name}_{suffix}.parquet';table.to_parquet(path,index=False);readback=pd.read_parquet(path);pd.testing.assert_frame_equal(table,readback,check_dtype=True,check_exact=True);hashes[suffix]=sha(path)
  result[name]={'rows':n,'cases_with_queries':ncase,'positive_hard_labels':pos,'predictor_columns_excluding_keys':68,'predeadline_only':early,'soft_target_in_supervision':name=='fit','readback_exact':True,'file_sha256':hashes}
 for a,b in [('fit','calibration'),('calibration','threshold_selection')]:
  need(not set(parts[a].case_seg)&set(parts[b].case_seg) and parts[a].t_next.max()+pd.Timedelta(hours=2)<parts[b].t0.min(),'STAGE_EMBARGO')
 r.update(stages=result,period_counts=counts,feature_order=COLS,frozen_pca_fit_rows=int(pca.n_samples_),oos_key_order_exact=True,stage_case_disjoint_and_chronological=True,soft_target_lambda=.25,soft_target_temperature=2.,all_input_pins_unchanged=all(sha(p[n])==h for n,h in DATA_PINS.items()) and sha(oos)==OOS_PIN and sha(pca_path)==PCA_PIN)
 need(r['all_input_pins_unchanged'],'INPUT_CHANGED');r['status']='MATRICES_PREPARED_PASS'
 if args.reference_matrices_dir:
  ref=Path(args.reference_matrices_dir); paths={}; checks={}
  for name in COUNTS:
   old=read_stage(ref,name,paths)
   for suffix,frame in zip(('predictors','supervision','retrospective_metadata'),old):
    new=pd.read_parquet(out/'matrices'/f'{name}_{suffix}.parquet')
    pd.testing.assert_frame_equal(new,frame,check_dtype=True,check_exact=True)
   checks[name]=True
  need(all(sha(p)==h for p,h in paths.items()),'REFERENCE_MATRIX_CHANGED')
  r['reference_matrices_exact']=checks
  r['status']='MATRICES_PREPARED_EXACT_REFERENCE_PASS'


def fit_or_validate(args,out,r):
 paths={};parts={k:read_stage(Path(args.matrices_dir),k,paths) for k in COUNTS}
 for a,b in [('fit','calibration'),('fit','threshold_selection'),('calibration','threshold_selection')]:
  need(set(parts[a][0].case_seg).isdisjoint(parts[b][0].case_seg),'CASE_OVERLAP')
  need(pd.to_datetime(parts[a][2].t_next,utc=True).max()+pd.Timedelta(hours=2)<pd.to_datetime(parts[b][2].t0,utc=True).min(),'EMBARGO')
 r.update(stage_counts={k:dict(rows=n,cases=c,positive=p) for k,(n,c,p) in COUNTS.items()},feature_count=len(COLS),training_performed=False,status='INPUT_VALIDATION_PASS')
 if args.command=='fit':
  fx,fs,_=parts['fit'];cx,cs,_=parts['calibration']
  model=fit_student(fx[COLS],fs.soft_target,fs.weight_predeadline)
  cal=calibrate(predict(model,cx[COLS]),cs.y_next_delay.to_numpy(),cs.weight_predeadline.to_numpy(np.float64))
  model.save_model(str(out/'student.json'));joblib.dump(cal,out/'calibrator.joblib',compress=3)
  r.update(training_performed=True,status='RECIPE_REFIT_COMPLETE_NOT_ARCHIVED_MODEL',fixed_thresholds=list(THRESHOLDS),asset_sha256={n:sha(out/n) for n in ('student.json','calibrator.joblib')})
 need(all(sha(p)==h for p,h in paths.items()),'INPUT_CHANGED');r['all_input_pins_unchanged']=True

def main():
 ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
 sub.add_parser('self-test')
 p=sub.add_parser('prepare');p.add_argument('--data-dir',required=True);p.add_argument('--teacher-export',required=True);p.add_argument('--pca',required=True);p.add_argument('--reference-matrices-dir');p.add_argument('--out-dir',required=True)
 for cmd in ('validate','fit'):
  p=sub.add_parser(cmd);p.add_argument('--matrices-dir',required=True);p.add_argument('--out-dir',required=True)
 args=ap.parse_args()
 if args.command=='self-test':matrix_selftest();training_selftest();return
 out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=False)
 r=dict(version=VERSION,status='BLOCKED',command=args.command,training_performed=False,teacher_or_pca_refit=False,threshold_search=False,official_TEST_or_VAL_read=False,original_assets_replaced=False,archived_model_equivalence_claimed=False,release_authorization_claimed=False)
 try:
  if args.command=='prepare':prepare(args,out,r)
  else:fit_or_validate(args,out,r)
 except Exception as e:r.update(status='BLOCKED',exception_type=type(e).__name__,check_code=str(e) if isinstance(e,GateError) else 'LOCAL_REVIEW_REQUIRED')
 (out/'summary.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf8');print(r['status'])
 if r['status']=='BLOCKED':raise SystemExit(1)
if __name__=='__main__':main()
