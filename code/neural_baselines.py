#!/usr/bin/env python3
"""Portable neural comparator recipes on the audited processed TRAIN inputs.

Validate reconstructs P4/P5 sequence inputs and the snapshot fit-only scaler;
no forward pass, fitting, threshold search or held-out access. Reproduce
optionally calibrates the frozen Transformer and sequence LSTM and trains a
separate T=1 snapshot LSTM. This is not training the two pretrained sequence
networks from scratch, and exact archived refits are not claimed. Use only
trusted authorized artifacts. Keep newly created weights and reports local
unless their disclosure is permitted. All outputs use a fresh directory.
"""
import argparse
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression

SIBLINGS={
 'primary_student.py':'a411d6dc5608efc3fd776c75bd239b1dd32b60f933d18e78fd6f756afd24a2f6',
 'teacher_oos.py':'5065757129830b3b2919ad0623611ab293aefc6e0d23fa792457274519c70c5a'}
for name,digest in SIBLINGS.items():
    p=Path(__file__).resolve().with_name(name)
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Verified companion modules must be alongside this script.')
import primary_student as core
import teacher_oos as oos
for m in (core,oos):
    core.need(Path(m.__file__).resolve()==Path(__file__).resolve().with_name(Path(m.__file__).name),'COMPANION_LOCATION')
THRESHOLDS={'standalone_transformer':(.23,.41),'sequence_lstm':(.26,.42),'snapshot_lstm':(.36,.67)}
ASSETS={
 'replay.py':'97f30c4dbe42f296d2410c867d6e8b2edeb19a7857219828c5203e4f732525ca',
 'sequence_lstm_preprocessing_contract.json':'440c8b37981c88a263f49500067dad963dc29b446a1d665508286c9813175f9b',
 'sequence_lstm.pt':'32352cf7a3d177dc8cbf2a231d4d9e2412f5f9c9772e0afe12a4e6f8bda16613',
 'snapshot_scaler.json':'22b6f88efa57197a769ab867c396c5a7a01a4c1a39bfe705599d92871697b200'}


def load_replay(path):
    core.pin(path,ASSETS['replay.py'])
    spec=importlib.util.spec_from_file_location('research_neural_replay',path)
    core.need(spec is not None and spec.loader is not None,'REPLAY_IMPORT')
    v=importlib.util.module_from_spec(spec);sys.modules[spec.name]=v;spec.loader.exec_module(v)
    core.need(v.BASE==core.BASE,'FEATURE_CONTRACT')
    return v


def snapshot_scaler(fit):
    a=fit[core.BASE].to_numpy(np.float32)
    core.need(np.isfinite(a).all(),'SNAPSHOT_FEATURES')
    z=np.sign(a)*np.log1p(np.abs(a))
    mu=z.mean(axis=0,dtype=np.float64).astype(np.float32)
    sd=z.std(axis=0,dtype=np.float64).astype(np.float32)
    sd=np.where(sd<1e-5,1.,sd).astype(np.float32)
    return {'features':core.BASE,'mu':mu.tolist(),'sd':sd.tolist(),'n_fitting_rows':len(fit),'n_fitting_cases':int(fit.case_seg.nunique()),'fitted_only_on':'TRAIN P0-P3 strictly predeadline','transform':'signlog + z-score, clip to [-12,12]'}


def compare_scalers(a,b):
    core.need(a['features']==b['features']==core.BASE and a['n_fitting_rows']==b['n_fitting_rows'] and a['n_fitting_cases']==b['n_fitting_cases'] and a['fitted_only_on']==b['fitted_only_on'],'SCALER_ROLE_OR_SCHEMA')
    result={k:core.compare(np.asarray(a[k],np.float32),np.asarray(b[k],np.float32)) for k in ('mu','sd')}
    return result,all(x['max_abs_difference']==0 for x in result.values())


def stage_queries(q,parts):
    result={}
    for stage in ('calibration','threshold_selection'):
        x,s,_=parts[stage]; keys=oos.keys(x[core.KEY]).reset_index(drop=True)
        z=keys.merge(q,on=core.KEY,how='left',sort=False,validate='one_to_one',indicator=True)
        core.need(z._merge.eq('both').all() and z[core.KEY].equals(keys) and z.part.eq('student_fit').all(),'SEQUENCE_STAGE_KEYS')
        result[stage]=z.drop(columns='_merge')
    return result


def prepare(args,v,paths):
    parts={n:core.read_stage(args.matrices_dir,n,paths) for n in core.STAGES}
    role_sets=[set(x.case_seg) for x,_,_ in parts.values()]
    core.need(all(not a&b for i,a in enumerate(role_sets) for b in role_sets[i+1:]),'STAGE_CASE_OVERLAP')
    d={n:core.pin(args.data_dir/(n+'.parquet'),h) for n,h in oos.PINS.items()};paths.update({p:oos.PINS[n] for n,p in d.items()})
    a={n:core.pin(args.bundle/n,h) for n,h in ASSETS.items()};paths.update({p:ASSETS[n] for n,p in a.items()})
    roles=pd.read_parquet(d['train_role_manifest'],columns=['case_seg','part'])
    inner=pd.read_parquet(d['teacher_inner_manifest'],columns=['case_seg','teacher_inner_role'])
    core.need(roles.case_seg.is_unique and inner.case_seg.is_unique,'ROLE_KEYS')
    sc=set(roles.loc[roles.part.eq('student_fit'),'case_seg']);tc=set(inner.case_seg);fc=set(inner.loc[inner.teacher_inner_role.eq('teacher_weights_fit'),'case_seg'])
    core.need(len(sc)==19845 and len(tc)==21461 and len(fc)==14272 and not sc&tc and tc==set(roles.loc[roles.part.eq('teacher_fit'),'case_seg']) and all(x<=sc for x in role_sets),'PRETRAINED_STUDENT_DISJOINT')
    pre=json.loads(a['sequence_lstm_preprocessing_contract.json'].read_text(encoding='utf-8'))
    core.need(pre['teacher_inner_case_digest']==hashlib.sha256('\n'.join(sorted(fc)).encode()).hexdigest(),'FIT_DIGEST')
    q=pd.read_parquet(d['train_query_index'],columns=core.KEY+['part','last_event_timestamp','observed_event_count','last_event_global_index']+v.QFEATURES[:2],filters=[('part','=','student_fit')]);q=oos.keys(q)
    core.need(len(q)==82786 and q.case_seg.nunique()==10311 and q.part.eq('student_fit').all(),'QUERY_COHORT')
    qstage=stage_queries(q,parts); tensors={}; report={}
    for stage,qs in qstage.items():
        e=pd.read_parquet(d['teacher_prefix_events'],columns=['case_seg',oos.TS,'concept:name']+v.EVENT_FEATURES,filters=[('case_seg','in',qs.case_seg.unique().tolist())]);core.need(set(e.case_seg)==set(qs.case_seg),'PREFIX_CASES')
        e,qs=oos.rebase(e,qs)
        evnum,qnum,evcat=v.apply_frozen_transform(e,qs,pre,v.teacher,14272)
        core.need((evcat!=v.UNK).all(),'UNKNOWN_TOKEN')
        # Verify prefix extraction at both ends of each stage, with dummy supervision.
        clean=qs.copy();clean['y_next_delay']=0.;clean['selected_weight']=1.
        ds=v.Queries(clean,evnum,qnum,evcat,100)
        for i in (0,len(ds)-1):
            cat,num,valid,_,_=ds[i];core.need(cat.shape==(100,) and num.shape==(100,26) and int(cat[-1])==v.QUERY and bool(valid[-1]) and float(num[-1,25])==1.,'PREFIX_TENSOR')
        tensors[stage]=(qs,evnum,qnum,evcat)
        report[stage]={'queries':len(qs),'cases':int(qs.case_seg.nunique()),'event_rows':len(e),'numeric_channels':26,'max_length':100,'unknown_tokens':int((evcat==v.UNK).sum()),'stage_key_order_preserved':True}
    scaler=snapshot_scaler(parts['fit'][0]);arch=json.loads(a['snapshot_scaler.json'].read_text(encoding='utf-8'));diff,exact=compare_scalers(scaler,arch)
    for stage,(x,_,_) in parts.items():
        z=v.transform(x,scaler,core.BASE);core.need(z.shape==(len(x),19) and np.isfinite(z).all(),'SNAPSHOT_TENSOR')
    return parts,tensors,scaler,report,diff,exact


def fit_cal(x,s):
    y=s.y_next_delay.to_numpy(np.int8);w=s.weight_predeadline.to_numpy(np.float64);x=np.asarray(x,np.float64).reshape(-1,1)
    core.need(len(x)==len(y)==len(w) and set(y)=={0,1} and np.isfinite(x).all() and np.isfinite(w).all() and (w>0).all(),'CALIBRATION')
    c=LogisticRegression(C=1000,solver='lbfgs',max_iter=1000,random_state=42);c.fit(x,y,sample_weight=w)
    core.need(np.isfinite(c.coef_).all() and np.isfinite(c.intercept_).all() and c.coef_[0,0]>0,'CALIBRATION_SLOPE');return c


def fit_snapshot(v,x,s,scaler,device,epochs=25,batch=512):
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    net=v.SnapshotLSTM(dim=19,hidden=128,dropout=.2).to(device);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    ds=TensorDataset(torch.from_numpy(v.transform(x,scaler,core.BASE)),torch.from_numpy(s.y_next_delay.to_numpy(np.float32)),torch.from_numpy(s.weight_predeadline.to_numpy(np.float32)))
    dl=DataLoader(ds,batch_size=batch,shuffle=True,num_workers=0,pin_memory=device.type=='cuda')
    history=[]
    for epoch in range(epochs):
        net.train();mass=loss_mass=0.;n=0
        for a,y,w in dl:
            a,y,w=a.to(device),y.to(device),w.to(device);z=net(a);loss=(torch.nn.functional.binary_cross_entropy_with_logits(z,y,reduction='none')*w).sum()/w.sum().clamp_min(1e-8)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            m=float(w.detach().sum().cpu());mass+=m;loss_mass+=float(loss.detach().cpu())*m;n+=len(y)
        core.need(n==len(x) and mass>0 and np.isfinite(loss_mass),'TRAINING_PASS');history.append(loss_mass/mass)
    return net.eval(),history


def reproduce(args,v,parts,tensors,scaler,device):
    xc,cs,_=parts['calibration'];xf,fs,_=parts['fit']
    # Standalone teacher calibration is on the previously regenerated logits.
    models={'standalone_transformer':fit_cal(xc.teacher_logit,cs)}
    ck=torch.load(core.pin(args.bundle/'sequence_lstm.pt',ASSETS['sequence_lstm.pt']),map_location='cpu',weights_only=True)
    core.need(ck['version']=='v4.124' and ck['kind']=='sequence' and ck['hidden']==128 and ck['max_len']==100,'SEQUENCE_CHECKPOINT')
    net=v.SequenceLSTM(ck['vocab_size'],hidden=128,dropout=.2);net.load_state_dict(ck['state_dict'],strict=True);net.to(device).eval()
    q,en,qn,ec=tensors['calibration'];raw=v.sequence_forward(net,q,en,qn,ec,device)
    models['sequence_lstm']=fit_cal(v.calibrated_raw_lr(raw),cs)
    snap,history=fit_snapshot(v,xf,fs,scaler,device)
    models['snapshot_lstm']=fit_cal(v.calibrated_raw_lr(v.snapshot_infer(snap,xc,scaler,core.BASE,device)),cs)
    for arm,cal in models.items():joblib.dump(cal,args.out_dir/(arm+'_calibrator.joblib'),compress=3)
    torch.save({'state_dict':{k:z.detach().cpu().clone() for k,z in snap.state_dict().items()},'kind':'snapshot19','train_role':'P0-P3 strictly predeadline','snapshot_epochs':25,'hidden':128,'num_layers':3,'dropout':.2,'features':core.BASE},args.out_dir/'snapshot_lstm.pt')
    (args.out_dir/'snapshot_scaler.json').write_text(json.dumps(scaler,indent=2)+'\n',encoding='utf-8')
    return history


def selftest(v):
    rng=np.random.default_rng(42);x=pd.DataFrame(rng.normal(size=(32,19)).astype(np.float32),columns=core.BASE);x['case_seg']=[str(i) for i in range(32)];s=pd.DataFrame({'y_next_delay':(x[core.BASE[0]]>0).astype(int),'weight_predeadline':np.ones(32)})
    sc=snapshot_scaler(x);core.need(compare_scalers(sc,sc)[1],'SCALER_SELF');core.need(v.transform(x,sc,core.BASE).shape==(32,19),'SCALER_TRANSFORM')
    torch.set_num_threads(1);net,h=fit_snapshot(v,x,s,sc,torch.device('cpu'),epochs=1,batch=16);p=v.snapshot_infer(net,x,sc,core.BASE,torch.device('cpu'));core.need(np.isfinite(p).all() and len(h)==1,'SNAPSHOT_FIT')
    # Synthetic calibration uses known informative scores, not incidental untrained logits.
    c=fit_cal(x[core.BASE[0]],s);core.need(c.coef_[0,0]>0,'PLATT_SELF')
    clean=pd.DataFrame({'case_seg':['a','b'],'k_idx':[1,1],'tau_ts':pd.to_datetime(['2025-01-01','2025-01-02'],utc=True),'last_event_global_index':[0,1],'observed_event_count':[1,1],'y_next_delay':[0.,0.],'selected_weight':[1.,1.]})
    ds=v.Queries(clean,np.zeros((2,23),np.float32),np.zeros((2,2),np.float32),np.array([4,4],np.int32),100)
    cat,num,valid,_,_=next(iter(DataLoader(ds,batch_size=2)));seq=v.SequenceLSTM(10);seq.eval()
    with torch.no_grad():z=seq(cat,num,valid)
    core.need(z.shape==(2,) and torch.isfinite(z).all(),'SEQUENCE_FORWARD_SELF')
    print('Synthetic snapshot training, scaler, calibration and sequence forward passed.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['validate','reproduce','self-test']);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--data-dir',type=Path);p.add_argument('--matrices-dir',type=Path);p.add_argument('--out-dir',type=Path);p.add_argument('--device',default='auto',choices=['auto','cpu','cuda']);args=p.parse_args()
    v=load_replay(args.bundle/'replay.py')
    if args.command=='self-test':selftest(v);return
    core.need(all(z is not None for z in (args.data_dir,args.matrices_dir,args.out_dir)),'REQUIRED_PATHS');core.need(not args.out_dir.exists(),'OUTPUT_EXISTS')
    paths={};parts,tensors,scaler,stages,diff,exact=prepare(args,v,paths)
    report={'version':'1.0','status':'NEURAL_INPUT_COMPATIBILITY_PASS' if exact else 'SNAPSHOT_SCALER_DIFFERENCE_REVIEW_REQUIRED','command':args.command,'sequence_stages':stages,'snapshot_stages':{n:{'rows':len(x),'features':19} for n,(x,_,_) in parts.items()},'snapshot_fit_scaler_comparison':diff,'snapshot_scaler_exact':exact,'fixed_thresholds':THRESHOLDS,'teacher_sequence_training_performed':False,'snapshot_training_performed':False,'calibrator_refit':False,'official_TEST_or_VAL_read':False,'threshold_search':False,'original_assets_replaced':False,'archived_model_equivalence_claimed':False,'release_authorization_claimed':False}
    args.out_dir.mkdir(parents=True,exist_ok=False)
    if args.command=='reproduce' and exact:
        device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
        report['snapshot_training_loss']=reproduce(args,v,parts,tensors,scaler,device);report.update(status='ISOLATED_NEURAL_RECIPE_COMPLETED_NO_ARCHIVED_PARITY_CLAIM',snapshot_training_performed=True,calibrator_refit=True)
    core.need(all(core.sha(p)==h for p,h in paths.items()),'INPUT_CHANGED');report['all_input_pins_unchanged']=True
    (args.out_dir/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(report['status'])

if __name__=='__main__':main()
