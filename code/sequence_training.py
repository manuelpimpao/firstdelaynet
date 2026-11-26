#!/usr/bin/env python3
"""Train fresh binary Transformer or sequence LSTM on processed TRAIN inputs.

Uses the fit-only frozen preprocessing contract, not original identifiers or
historical raw data. Validation checks input roles, weighted supervision and
prefix tensors without fitting or scoring a model. Optional fit initializes
new network weights and runs the archived weighted-BCE/early-stop recipe;
not a bitwise recreation of archived weights or original random-state history.
The output is separate, not a replacement for the published frozen assets.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

COMPANIONS = {'teacher_oos.py':'5065757129830b3b2919ad0623611ab293aefc6e0d23fa792457274519c70c5a',
              'primary_student.py':'a411d6dc5608efc3fd776c75bd239b1dd32b60f933d18e78fd6f756afd24a2f6',
              'neural_baselines.py':'3857c775dbc11c79e3dcd50a5c9cce1349945ed9d6d3fadf6b70f5859ad12516'}
for name,digest in COMPANIONS.items():
    p=Path(__file__).resolve().with_name(name)
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Verified companion scripts must be alongside this script.')
import teacher_oos as oos
import primary_student as core
import neural_baselines as neural
for m in (oos,core,neural):
    core.need(Path(m.__file__).resolve()==Path(__file__).resolve().with_name(Path(m.__file__).name),'COMPANION_LOCATION')
ROLE_COUNTS={'teacher_weights_fit':67471,'teacher_early_stop':12313,'PURGED_INNER':12896}
CONFIG={'transformer':dict(epochs=12,patience=3,lr=.0003,seed=42,batch=128),
        'sequence_lstm':dict(epochs=20,patience=5,lr=.0002,seed=42,batch=128)}

def seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)

def validate_supervision(q):
    core.need(q.y_next_delay.isin([0,1]).all(),'BINARY_LABELS')
    w=q.selected_weight.to_numpy(float)
    core.need(np.isfinite(w).all() and (w>0).all(),'POSITIVE_WEIGHT')
    core.need(np.allclose(q.groupby(['case_seg','k_idx'],observed=True).selected_weight.sum().to_numpy(),1.,atol=1e-9,rtol=0),'INTERVAL_MASS')
    for n in ('teacher_weights_fit','teacher_early_stop'):
        core.need(q.loc[q.inner_role.eq(n),'y_next_delay'].nunique()==2,'ROLE_CLASS_SUPPORT')

def datasets(v,q,evnum,qnum,evcat):
    ds={}
    for role in ('teacher_weights_fit','teacher_early_stop'):
        ix=np.flatnonzero(q.inner_role.eq(role).to_numpy())
        subset=q.iloc[ix].reset_index(drop=True)
        ds[role]=v.Queries(subset,evnum,qnum[ix],evcat,100)
        for i in (0,len(subset)-1):
            c,x,valid,y,w=ds[role][i]
            core.need(c.shape==(100,) and x.shape==(100,26) and bool(valid[-1]) and int(c[-1])==v.QUERY and float(x[-1,25])==1.,'QUERY_TENSOR')
            core.need(torch.isfinite(x).all() and float(y)==float(subset.iloc[i].y_next_delay) and np.isclose(float(w),float(subset.iloc[i].selected_weight)),'TENSOR_SUPERVISION')
    return ds

def prepare(args,v,paths):
    d={n:core.pin(args.data_dir/(n+'.parquet'),h) for n,h in oos.PINS.items()}
    d['train_query_labels']=core.pin(args.data_dir/'train_query_labels.parquet',core.DATA_PINS['train_query_labels'])
    paths.update({p:oos.PINS[n] if n in oos.PINS else core.DATA_PINS[n] for n,p in d.items()})
    prepath=core.pin(args.bundle/'teacher_preprocessing_contract.json',oos.ASSETS['teacher_preprocessing_contract.json'])
    seqpath=core.pin(args.bundle/'sequence_lstm_preprocessing_contract.json',neural.ASSETS['sequence_lstm_preprocessing_contract.json'])
    paths[prepath]=oos.ASSETS['teacher_preprocessing_contract.json'];paths[seqpath]=neural.ASSETS['sequence_lstm_preprocessing_contract.json']
    pre=json.loads(prepath.read_text(encoding='utf8'));core.need(pre==json.loads(seqpath.read_text(encoding='utf8')),'TRANSFORM_CONTRACT_DIFFERENCE')
    outer=pd.read_parquet(d['train_role_manifest'],columns=['case_seg','part']);inner=pd.read_parquet(d['teacher_inner_manifest'],columns=['case_seg','teacher_inner_role'])
    core.need(outer.case_seg.is_unique and inner.case_seg.is_unique and len(outer)==82930 and len(inner)==21461,'MANIFEST_KEYS')
    tc=set(inner.case_seg);fc=set(inner.loc[inner.teacher_inner_role.eq('teacher_weights_fit'),'case_seg']);sc=set(outer.loc[outer.part.eq('student_fit'),'case_seg'])
    core.need(tc==set(outer.loc[outer.part.eq('teacher_fit'),'case_seg']) and not tc&sc and len(fc)==14272 and set(inner.teacher_inner_role)==set(ROLE_COUNTS),'CASE_ROLES')
    core.need(pre['teacher_inner_case_digest']==hashlib.sha256('\n'.join(sorted(fc)).encode()).hexdigest(),'FIT_DIGEST')
    q=pd.read_parquet(d['train_query_index'],columns=core.KEY+['part','last_event_timestamp','observed_event_count','last_event_global_index']+v.QFEATURES[:2],filters=[('part','=','teacher_fit')]);q=core.keyed(q)
    core.need(len(q)==92680 and q.case_seg.nunique()==11651 and q.part.eq('teacher_fit').all() and set(q.case_seg)<=tc,'TEACHER_QUERY_COHORT')
    # Read only query-bearing teacher cases; no student, assessment or held-out outcomes.
    y=core.keyed(pd.read_parquet(d['train_query_labels'],columns=core.KEY+['y_next_delay','weight_full_clock'],filters=[('case_seg','in',q.case_seg.unique().tolist())]))
    old=q[core.KEY].reset_index(drop=True)
    q=q.merge(y,on=core.KEY,how='left',sort=False,validate='one_to_one',indicator=True)
    core.need(q._merge.eq('both').all() and q[core.KEY].equals(old),'SUPERVISION_ALIGNMENT');q=q.drop(columns='_merge')
    q['inner_role']=q.case_seg.map(inner.set_index('case_seg').teacher_inner_role);q['selected_weight']=q.weight_full_clock.astype(float)
    core.need(q.inner_role.value_counts().to_dict()==ROLE_COUNTS and int(q.y_next_delay.sum())==45578,'ROLE_OR_LABEL_COUNTS')
    validate_supervision(q)
    e=pd.read_parquet(d['teacher_prefix_events'],columns=['case_seg',oos.TS,'concept:name']+v.EVENT_FEATURES,filters=[('case_seg','in',sorted(tc))])
    core.need(set(e.case_seg)==tc,'EVENT_CASES')
    e,q=oos.rebase(e,q)
    evnum,qnum,evcat=v.apply_frozen_transform(e,q,pre,v.teacher,14272)
    core.need((evcat!=v.UNK).all(),'UNKNOWN_TOKEN')
    # Recompute fit-only numerical statistics for documentation, not replacement.
    em=e.case_seg.isin(fc).to_numpy();qm=q.inner_role.eq('teacher_weights_fit').to_numpy()
    z=v.signlog(e[v.EVENT_FEATURES].to_numpy(np.float32));qr=np.log1p(np.stack([q[n].to_numpy(np.float32) for n in v.QFEATURES[:2]],axis=1))
    def stats(x,mask):
        mu=x[mask].mean(axis=0,dtype=np.float64).astype(np.float32);sd=x[mask].std(axis=0,dtype=np.float64).astype(np.float32)
        return mu,np.where(sd<1e-5,1.,sd).astype(np.float32)
    emu,esd=stats(z,em);qmu,qsd=stats(qr,qm)
    scaler={k:core.compare(a,np.asarray(pre[k],np.float32)) for k,a in zip(('event_mean','event_std','query_log1p_mean','query_log1p_std'),(emu,esd,qmu,qsd))}
    observed=set(e.loc[em,'concept:name'].astype(str));core.need(observed==set(pre['vocab'])-set(v.teacher.SPECIAL),'FIT_VOCAB_MEMBERSHIP')
    ds=datasets(v,q,evnum,qnum,evcat)
    report={'query_rows':len(q),'event_rows':len(e),'event_cases':len(tc),'fit_scaler_cases':len(fc),'role_counts':ROLE_COUNTS,'positive_labels':int(q.y_next_delay.sum()),'fit_event_rows':int(em.sum()),'unknown_tokens':0,'fit_only_scaler_comparison':scaler,'frozen_preprocessing_used':True,'fit_vocab_membership_exact':True,'query_input_numeric_channels':26,'max_length':100,'role_case_disjoint':True,'unit_interval_weights':True,'label_input_separation':True}
    return ds,pre,report

def epoch(net,loader,device,kind,opt=None):
    net.train(opt is not None);total=mass=0.
    for cat,num,valid,y,w in loader:
        cat,num,valid,y,w=(x.to(device) for x in (cat,num,valid,y,w))
        with torch.set_grad_enabled(opt is not None):
            p=net(cat,num,valid);p=p[0] if kind=='transformer' else p
            loss=(nn.functional.binary_cross_entropy_with_logits(p,y,reduction='none')*w).sum()/w.sum().clamp_min(1e-8)
            core.need(torch.isfinite(loss),'NONFINITE_LOSS')
            if opt is not None:
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        m=float(w.sum().detach().cpu());mass+=m;total+=float(loss.detach().cpu())*m
    core.need(mass>0,'EMPTY_ROLE');return total/mass

def fit(v,ds,pre,kind,device,out,synthetic=False):
    cfg=CONFIG[kind];seed(cfg['seed'])
    if kind=='transformer':
        if device.type=='cuda':torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        net=v.Teacher(pre['vocab_size'],d_model=192,nhead=4,layers=2,dropout=.1,max_len=100)
        gen=torch.Generator().manual_seed(cfg['seed']+119)
    else:
        net=v.SequenceLSTM(pre['vocab_size'],hidden=128,dropout=.2);gen=None
    net=net.to(device)
    train=DataLoader(ds['teacher_weights_fit'],batch_size=cfg['batch'],shuffle=True,generator=gen,num_workers=0,pin_memory=device.type=='cuda')
    stop=DataLoader(ds['teacher_early_stop'],batch_size=cfg['batch'],shuffle=False,num_workers=0,pin_memory=device.type=='cuda')
    opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=.01)
    if kind=='transformer':seed(cfg['seed']+771)
    best=float('inf');state=None;wait=0;history=[];best_ep=None
    for i in range(1,(1 if synthetic else cfg['epochs'])+1):
        a=epoch(net,train,device,kind,opt);b=epoch(net,stop,device,kind)
        history.append({'epoch':i,'fit_weighted_BCE':a,'early_stop_weighted_BCE':b});print(kind,i,a,b,flush=True)
        if b<best-1e-5:
            best=b;wait=0;best_ep=i;state={k:x.detach().cpu().clone() for k,x in net.state_dict().items()}
        else:
            wait+=1
            if wait>=cfg['patience']:break
    core.need(state is not None,'NO_CHECKPOINT')
    dest=out/(kind+'_fresh.pt');torch.save({'state_dict':state,'kind':kind,'vocab_size':pre['vocab_size'],'best_epoch':best_ep,'max_len':100},dest)
    rb=torch.load(dest,map_location='cpu',weights_only=True);core.need(all(torch.equal(x,rb['state_dict'][k]) for k,x in state.items()),'CHECKPOINT_READBACK')
    return {'best_epoch':best_ep,'epochs':len(history),'early_stop_not_independent':True,'history':history,'checkpoint_sha256':core.sha(dest)}

def selftest(v):
    import tempfile
    torch.set_num_threads(1)
    t=pd.Timestamp('2025-01-01',tz='UTC')
    q=pd.DataFrame({'case_seg':['a','b','c','d'],'k_idx':[1]*4,'tau_ts':[t]*4,'last_event_global_index':np.arange(4),'observed_event_count':[1]*4,'y_next_delay':[0,1,0,1],'selected_weight':[1.]*4,'inner_role':['teacher_weights_fit']*2+['teacher_early_stop']*2})
    validate_supervision(q);ds=datasets(v,q,np.zeros((4,23),np.float32),np.zeros((4,2),np.float32),np.full(4,3,np.int32))
    with tempfile.TemporaryDirectory() as d:
        for kind in CONFIG:fit(v,ds,{'vocab_size':4},kind,torch.device('cpu'),Path(d),synthetic=True)
    q.loc[0,'selected_weight']=2.
    try:validate_supervision(q)
    except core.GateError:pass
    else:raise RuntimeError('SYNTHETIC_WEIGHT_GUARD')
    print('Synthetic weighted supervision, prefix tensor, two neural fits and checkpoint readback passed.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['validate','fit','self-test']);p.add_argument('--data-dir',type=Path);p.add_argument('--bundle',required=True,type=Path);p.add_argument('--out-dir',type=Path);p.add_argument('--model',choices=['transformer','sequence_lstm','both'],default='both');p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    v=neural.load_replay(a.bundle/'replay.py')
    if a.command=='self-test':selftest(v);return
    core.need(a.data_dir is not None and a.out_dir is not None,'REQUIRED_PATHS');a.out_dir.mkdir(parents=True,exist_ok=False)
    paths={a.bundle/'replay.py':neural.ASSETS['replay.py']}
    r={'version':'1.0','status':'BLOCKED','command':a.command,'training_performed':False,'official_TEST_or_VAL_read':False,'threshold_search':False,'archived_model_equivalence_claimed':False,'original_assets_replaced':False,'release_authorization_claimed':False}
    try:
        ds,pre,rep=prepare(a,v,paths);r.update(rep)
        if a.command=='fit':
            core.need(a.device!='cuda' or torch.cuda.is_available(),'CUDA_UNAVAILABLE');r['training_performed']=True;r['fresh_models']={}
            for kind in (CONFIG if a.model=='both' else (a.model,)):r['fresh_models'][kind]=fit(v,ds,pre,kind,torch.device(a.device),a.out_dir)
        r['all_input_pins_unchanged']=all(core.sha(p)==h for p,h in paths.items());core.need(r['all_input_pins_unchanged'],'INPUT_CHANGED')
        r['status']='SEQUENCE_TRAIN_INPUT_COMPATIBILITY_PASS' if a.command=='validate' else 'FRESH_SEQUENCE_TRAINING_COMPLETE_NOT_ARCHIVE_EQUIVALENCE'
    except Exception as e:r.update(exception_type=type(e).__name__,check_code=str(e) if isinstance(e,(core.GateError,oos.Blocked)) else 'LOCAL_REVIEW_REQUIRED')
    (a.out_dir/'summary.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf8');print(r['status']);raise SystemExit(int(r['status']=='BLOCKED'))
if __name__=='__main__':main()
