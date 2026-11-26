#!/usr/bin/env python3
"""Portable TRAIN-only export of frozen binary teacher out-of-sample features.
Uses processed research inputs and the frozen replay module and checkpoint.
No fitting, labels, held-out data, threshold changes or original writes.
Reference comparison is optional and uses only the transformed export.
Use trusted, authorized artifacts; do not redistribute row-level outputs
unless covered by the applicable release permissions.
"""
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
VERSION='1.0'
KEY=['case_seg','k_idx','tau_ts']; TS='time:timestamp'
PINS={'teacher_prefix_events':'867b02be549d524fba0a42f5b748e5a5c5b27d75379b694ded743196beeb9cca','train_query_index':'81a78054d1980de1c1bd888d459a7b773d87191a79412ce140c097a503dd7cbe','train_role_manifest':'bae762c66201da8253bedd39e073a74879a6c36a7b498efc332f110eac37beb2','teacher_inner_manifest':'6e2c1d76df73b8befec3da4b14041a9e19890ed543fa686caea93aa718f0b424'}
ASSETS={'replay.py':'97f30c4dbe42f296d2410c867d6e8b2edeb19a7857219828c5203e4f732525ca','teacher.pt':'510ca342c9e7afd4a54cafb39c735080d35954e641ae0669ceb1b029268c3bb8','teacher_preprocessing_contract.json':'440c8b37981c88a263f49500067dad963dc29b446a1d665508286c9813175f9b'}
REFERENCE_PIN='c7bb0f9110c084f92c0118df59ec1335687c64963c7c241c6f486ea6c9b1c3a8'
FEATURES=['teacher_logit']+[f'teacher_embed_{i:03d}' for i in range(192)]
TOL=1e-6
class Blocked(Exception):pass
def need(ok,code):
 if not bool(ok):raise Blocked(code)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def pin(p,h):
 need(p.is_file() and not p.is_symlink() and sha(p)==h,'PIN_MISMATCH');return p
def utc(s):return pd.to_datetime(s,utc=True).astype('datetime64[ns, UTC]')
def keys(d):
 d=d.copy();d['case_seg']=d.case_seg.astype('string');d['tau_ts']=utc(d.tau_ts)
 need(d[KEY].notna().all().all() and not d.duplicated(KEY).any(),'INVALID_QUERY_KEYS');return d
def rebase(e,q):
 e=e.copy();q=keys(q);e['case_seg']=e.case_seg.astype('string');e[TS]=utc(e[TS]);q['last_event_timestamp']=utc(q.last_event_timestamp)
 need(e[['case_seg',TS]].notna().all().all() and not e.duplicated(['case_seg',TS]).any(),'EVENT_KEYS')
 e=e.sort_values(['case_seg',TS],kind='mergesort').reset_index(drop=True)
 m=e[['case_seg',TS]].copy();m['pos']=np.arange(len(m),dtype=np.int64);m['start']=m.groupby('case_seg',observed=True).pos.transform('min')
 old=q[KEY].copy();q=q.merge(m,left_on=['case_seg','last_event_timestamp'],right_on=['case_seg',TS],how='left',sort=False,validate='many_to_one',indicator=True)
 need(q._merge.eq('both').all() and q[KEY].equals(old.reset_index(drop=True)),'ENDPOINT_OR_ORDER')
 pos=q.pos.to_numpy(np.int64);count=q.observed_event_count.to_numpy(np.int64)
 need(np.array_equal(pos-q.start.to_numpy(np.int64)+1,count) and (count>=1).all(),'PREFIX_COUNT')
 need(q.last_event_timestamp.le(q.tau_ts).all(),'FUTURE_ENDPOINT');nxt=np.minimum(pos+1,len(e)-1)
 same=e.case_seg.to_numpy()[nxt]==q.case_seg.to_numpy();has_next=pos+1<len(e)
 need(np.all(~(same&has_next)|(e[TS].array.asi8[nxt]>q.tau_ts.array.asi8)),'OMITTED_VISIBLE_EVENT')
 need(np.allclose((q.tau_ts-q.last_event_timestamp).dt.total_seconds(),q.query_seconds_since_last_event,atol=1e-6,rtol=0),'LAST_EVENT_GAP')
 q['last_event_global_index']=pos
 return e,q.drop(columns=[TS,'pos','start','_merge'])
def differences(a,b):
 need(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),'NONFINITE_OR_SHAPE')
 x=np.abs(a.astype(np.float64)-b.astype(np.float64));return {'values':int(x.size),'exact_values':int((x==0).sum()),'max_abs_difference':float(x.max()),'mean_abs_difference':float(x.mean()),'values_above_1e_6':int((x>TOL).sum()),'rows_above_1e_6':int((x.reshape(len(a),-1).max(axis=1)>TOL).sum())}
def selftest():
 t=pd.Timestamp('2025-01-01',tz='UTC');e=pd.DataFrame({'case_seg':['b','a','a'],TS:[t,t,t+pd.Timedelta(minutes=1)]});q=pd.DataFrame({'case_seg':['b','a'],'k_idx':[1,1],'tau_ts':[t,t+pd.Timedelta(minutes=2)],'last_event_timestamp':[t,t+pd.Timedelta(minutes=1)],'observed_event_count':[1,2],'query_seconds_since_last_event':[0.,60.]})
 _,z=rebase(e,q);need(z.last_event_global_index.tolist()==[2,1],'SYNTHETIC_REBASE')
 q.loc[1,'last_event_timestamp']=t;q.loc[1,'observed_event_count']=1;q.loc[1,'query_seconds_since_last_event']=120
 try:rebase(e,q)
 except Blocked as ex:need(str(ex)=='OMITTED_VISIBLE_EVENT','SYNTHETIC_WRONG_GUARD')
 else:raise Blocked('SYNTHETIC_FUTURE_GUARD')
 need(differences(np.zeros((2,3)),np.zeros((2,3)))['exact_values']==6,'SYNTHETIC_PARITY');print('Synthetic endpoint, omitted-event and numerical tests passed.')
def run(draft,bundle,reference,out,r,device):
 p={n:pin(draft/(n+'.parquet'),h) for n,h in PINS.items()};a={n:pin(bundle/n,h) for n,h in ASSETS.items()}
 if reference is not None:pin(reference,REFERENCE_PIN)
 spec=importlib.util.spec_from_file_location('research_frozen_replay',a['replay.py']);v=importlib.util.module_from_spec(spec);sys.modules[spec.name]=v;spec.loader.exec_module(v)
 roles=pd.read_parquet(p['train_role_manifest'],columns=['case_seg','part']);inner=pd.read_parquet(p['teacher_inner_manifest'],columns=['case_seg','teacher_inner_role']);need(roles.case_seg.is_unique and inner.case_seg.is_unique,'ROLE_KEYS')
 sc=set(roles.loc[roles.part.eq('student_fit'),'case_seg']);tc=set(inner.case_seg);fc=set(inner.loc[inner.teacher_inner_role.eq('teacher_weights_fit'),'case_seg']);need(len(sc)==19845 and len(tc)==21461 and len(fc)==14272 and not sc&tc and tc==set(roles.loc[roles.part.eq('teacher_fit'),'case_seg']),'ROLE_DISJOINTNESS')
 pre=json.loads(a['teacher_preprocessing_contract.json'].read_text(encoding='utf8'));need(pre['teacher_inner_case_digest']==hashlib.sha256('\n'.join(sorted(fc)).encode()).hexdigest(),'FIT_DIGEST')
 q=pd.read_parquet(p['train_query_index'],columns=KEY+['part','last_event_timestamp','observed_event_count','last_event_global_index']+v.QFEATURES[:2],filters=[('part','=','student_fit')]);need(len(q)==82786 and q.case_seg.nunique()==10311 and q.part.eq('student_fit').all() and set(q.case_seg)<=sc,'OOS_QUERY_COHORT')
 e=pd.read_parquet(p['teacher_prefix_events'],columns=['case_seg',TS,'concept:name']+v.EVENT_FEATURES,filters=[('case_seg','in',q.case_seg.unique().tolist())]);need(set(e.case_seg)==set(q.case_seg),'OOS_PREFIX_CASES')
 e,q=rebase(e,q);evnum,qnum,evcat=v.apply_frozen_transform(e,q,pre,v.teacher,14272);need((evcat!=v.UNK).all(),'UNKNOWN_TOKEN')
 ck=torch.load(a['teacher.pt'],map_location='cpu',weights_only=True);need(ck['version']=='v4.194' and ck['arm']=='binary_control','CHECKPOINT_LINEAGE');cfg=ck['teacher_cfg'];need(cfg['vocab_size']==pre['vocab_size'],'VOCAB_SIZE');net=v.Teacher(**cfg);net.load_state_dict(ck['model_state'],strict=True);net.to(device).eval()
 # Keep candidate query row order; original V4.195 uses index row order, not lexical case sort for batching.
 print('Regenerating frozen binary teacher TRAIN OOS features...',flush=True)
 logits,emb=v.teacher_infer(v.teacher,net,'binary_control',q,evnum,qnum,evcat,128,device,cfg['max_len']);need(emb.shape==(82786,192),'EMBEDDING_SHAPE')
 pred=np.column_stack((logits,emb)).astype(np.float32);need(np.isfinite(pred).all(),'NONFINITE_OUTPUT')
 qkey=keys(q[KEY]).reset_index(drop=True)
 result=pd.concat([qkey,pd.DataFrame(pred,columns=FEATURES)],axis=1)
 r.update(queries=len(q),query_cases=int(q.case_seg.nunique()),event_prefix_rows=len(e),embedding_dimensions=192,teacher_student_cases_disjoint=True,unknown_tokens=0,device=str(device),torch_version=str(torch.__version__),batch_size=128)
 if reference is not None:
  ref=pd.read_parquet(reference,columns=KEY+FEATURES);ref=keys(ref)
  need(len(ref)==len(q),'REFERENCE_ROWS')
  r['reference_query_order_exact']=qkey.equals(ref[KEY].reset_index(drop=True));need(r['reference_query_order_exact'],'REFERENCE_ORDER')
  refvals=ref[FEATURES].to_numpy(np.float32)
  r['logit_parity']=differences(pred[:,0],refvals[:,0]);r['embedding_parity']=differences(pred[:,1:],refvals[:,1:])
  r['reference_values_exact']=r['logit_parity']['max_abs_difference']==0 and r['embedding_parity']['max_abs_difference']==0
  need(r['reference_values_exact'],'REFERENCE_VALUE_DIFFERENCE')
 dest=out/'teacher_oos.parquet';result.to_parquet(dest,index=False);rb=pd.read_parquet(dest)
 need(keys(rb)[KEY].equals(qkey) and np.array_equal(rb[FEATURES].to_numpy(np.float32),pred),'OUTPUT_READBACK')
 r['export_sha256']=sha(dest);r['reference_byte_fingerprint_exact']=r['export_sha256']==REFERENCE_PIN
 # The companion primary_student.py accepts the pinned verified export bytes.
 # Equivalent Parquet encodings may differ by writer/environment; do not weaken its gate.
 r['all_input_pins_unchanged']=all(sha(p[n])==h for n,h in PINS.items()) and all(sha(a[n])==h for n,h in ASSETS.items()) and (reference is None or sha(reference)==REFERENCE_PIN)
 need(r['all_input_pins_unchanged'],'INPUT_CHANGED')
 r['status']='OOS_EXACT_REFERENCE_PASS' if reference is not None and r['reference_byte_fingerprint_exact'] else ('OOS_VALUES_EXACT_ENCODING_DIFFERS' if reference is not None else 'OOS_GENERATED_REFERENCE_NOT_COMPARED')
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--data-dir',type=Path);ap.add_argument('--bundle',type=Path);ap.add_argument('--reference-export',type=Path);ap.add_argument('--out-dir',type=Path);ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');ap.add_argument('--self-test',action='store_true');args=ap.parse_args();selftest()
 if args.self_test:return
 need(args.data_dir is not None and args.bundle is not None and args.out_dir is not None,'REQUIRED_PATHS')
 need(args.device!='cuda' or torch.cuda.is_available(),'CUDA_UNAVAILABLE')
 out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=False)
 r={'version':VERSION,'status':'BLOCKED','scope':'processed TRAIN inputs; frozen teacher export only','official_TEST_read':False,'official_VAL_read':False,'labels_read':False,'training_performed':False,'threshold_changes':False,'release_authorization_claimed':False,'reference_compared':args.reference_export is not None}
 try:run(args.data_dir.resolve(),args.bundle.resolve(),args.reference_export.resolve() if args.reference_export else None,out,r,torch.device(args.device))
 except Exception as exc:r.update(exception_type=type(exc).__name__,check_code=str(exc) if isinstance(exc,Blocked) else 'LOCAL_REVIEW_REQUIRED')
 (out/'summary.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf8');print(r['status']);raise SystemExit(0 if r['status'] in ('OOS_EXACT_REFERENCE_PASS','OOS_GENERATED_REFERENCE_NOT_COMPARED') else 1)
if __name__=='__main__':main()
