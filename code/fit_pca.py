#!/usr/bin/env python3
"""Fit-only PCA recipe for the processed teacher–student research inputs.
Uses the verified label-free teacher export and the existing chronological
student matrices. Fits only the 192-to-48 PCA on predeadline P0–P3 rows;
P4 and P5 are transformed, never fitted. No prediction model or threshold
is fitted. The frozen reference remains unchanged. Use trusted pinned assets.
A fresh PCA with different numerical results is not a replacement model asset.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.decomposition import PCA

COMPANION_SHA = 'a411d6dc5608efc3fd776c75bd239b1dd32b60f933d18e78fd6f756afd24a2f6'
TOL = 1e-6


def require(ok, code):
    if not bool(ok):
        raise ValueError(code)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def companion():
    p = Path(__file__).resolve().with_name('primary_student.py')
    require(p.is_file() and sha(p) == COMPANION_SHA, 'COMPANION_PIN')
    import primary_student as core
    require(Path(core.__file__).resolve() == p, 'COMPANION_LOCATION')
    return core


def fit_only(x):
    require(x.ndim == 2 and x.shape[1] == 192 and len(x) > 48
            and x.dtype == np.float32 and np.isfinite(x).all(), 'PCA_FIT_INPUT')
    model = PCA(n_components=48, svd_solver='randomized', random_state=42)
    model.fit(x)
    require(model.n_samples_ == len(x) and np.isfinite(model.components_).all(), 'PCA_FIT_OUTPUT')
    return model


def aligned_embeddings(keys, export, core):
    keys = core.keyed(keys[core.KEY])
    z = keys.merge(export[core.KEY+core.EMBED], on=core.KEY, how='left', sort=False,
                   validate='one_to_one', indicator=True)
    require(z._merge.eq('both').all() and z[core.KEY].equals(keys.reset_index(drop=True)), 'EMBEDDING_ALIGNMENT')
    x = z[core.EMBED].to_numpy(np.float32)
    require(x.shape == (len(keys),192) and np.isfinite(x).all(), 'EMBEDDING_VALUES')
    return x


def compare(a,b):
    a,b = np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    require(a.shape == b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all(), 'COMPARE')
    d = np.abs(a-b)
    return dict(values=int(d.size), exact_values=int((d==0).sum()), max_abs_difference=float(d.max()),
                mean_abs_difference=float(d.mean()), values_above_1e_6=int((d>TOL).sum()))


def self_test(core):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(150,192)).astype(np.float32)
    t = pd.Timestamp('2025-01-01',tz='UTC')
    keys = core.keyed(pd.DataFrame({'case_seg':[f'c{i}' for i in range(len(x))], 'k_idx':1, 'tau_ts':t}))
    export = pd.concat([keys,pd.DataFrame(x,columns=core.EMBED)],axis=1)
    a = aligned_embeddings(keys.iloc[:100],export,core)
    b = aligned_embeddings(keys.iloc[100:],export,core)
    require(np.array_equal(a,x[:100]) and np.array_equal(b,x[100:]),'SYNTHETIC_ORDER')
    model = fit_only(a)
    y = model.transform(b)
    require(y.shape==(50,48) and model.n_samples_==100,'SYNTHETIC_TRANSFORM')
    require(compare(y,y)['values_above_1e_6']==0,'SYNTHETIC_EQUAL')
    require(compare(y,y+.01)['values_above_1e_6']==y.size,'SYNTHETIC_DIFFERENCE')
    try:
        aligned_embeddings(keys.iloc[:100],export.iloc[1:],core)
    except ValueError:
        pass
    else:
        raise AssertionError('SYNTHETIC_MISSING_KEY_ACCEPTED')
    print('Synthetic PCA fit, split alignment, comparison and missing-key guards passed.')


def run(args, out, r, core):
    paths = {}
    stages = {n:core.read_stage(Path(args.matrices_dir),n,paths) for n in core.COUNTS}
    for a,b in [('fit','calibration'),('fit','threshold_selection'),('calibration','threshold_selection')]:
        require(set(stages[a][0].case_seg).isdisjoint(stages[b][0].case_seg),'CASE_OVERLAP')
        require(pd.to_datetime(stages[a][2].t_next,utc=True).max()+pd.Timedelta(hours=2)
                <pd.to_datetime(stages[b][2].t0,utc=True).min(),'EMBARGO')
    ep = core.pin(Path(args.teacher_export),core.OOS_PIN); paths[ep]=core.OOS_PIN
    fp = core.pin(Path(args.frozen_pca),core.PCA_PIN); paths[fp]=core.PCA_PIN
    export=core.keyed(pd.read_parquet(ep,columns=core.KEY+core.EMBED))
    require(len(export)==82786 and export.case_seg.nunique()==10311,'OOS_CENSUS')
    # Use the exact validated row order from the portable primary matrices.
    xs = {n:aligned_embeddings(v[0],export,core) for n,v in stages.items()}
    frozen=joblib.load(fp)
    require(frozen.n_samples_==31080 and frozen.n_features_in_==192 and frozen.n_components_==48
            and frozen.svd_solver=='randomized' and frozen.random_state==42,'FROZEN_PCA_CONTRACT')
    frozen_comparisons={n:compare(frozen.transform(x).astype(np.float32),stages[n][0][core.PC].to_numpy(np.float32)) for n,x in xs.items()}
    r['frozen_transform_vs_reference_matrices']=frozen_comparisons
    require(all(v['values_above_1e_6']==0 for v in frozen_comparisons.values()),'FROZEN_TRANSFORM_PARITY')
    r.update(stage_rows={n:len(x) for n,x in xs.items()}, fit_cases=stages['fit'][0].case_seg.nunique(),
             components=48, embedding_dimensions=192, svd_solver='randomized', random_state=42,
             fit_dtype=str(xs['fit'].dtype), fit_c_contiguous=bool(xs['fit'].flags.c_contiguous),
             fit_f_contiguous=bool(xs['fit'].flags.f_contiguous),status='PCA_INPUTS_VALIDATED')
    if args.command=='fit':
        model=fit_only(xs['fit']);r['pca_fitted']=True
        r['parameter_comparisons']={k:compare(getattr(model,k),getattr(frozen,k)) for k in ('components_','mean_','explained_variance_','explained_variance_ratio_','singular_values_')}
        r['fresh_transform_vs_frozen']={n:compare(model.transform(x),frozen.transform(x)) for n,x in xs.items()}
        passed=all(v['values_above_1e_6']==0 for v in list(r['parameter_comparisons'].values())+list(r['fresh_transform_vs_frozen'].values()))
        dest=out/'pca_refit.joblib';joblib.dump(model,dest,compress=3)
        rb=joblib.load(dest)
        r['serialization_parameter_comparisons']={k:compare(getattr(model,k),getattr(rb,k)) for k in ('components_','mean_','explained_variance_','explained_variance_ratio_','singular_values_')}
        r['serialization_transform_comparison']=compare(model.transform(xs['fit'][:128]),rb.transform(xs['fit'][:128]))
        serialized_exact=all(z['max_abs_difference']==0 for z in list(r['serialization_parameter_comparisons'].values())+[r['serialization_transform_comparison']])
        r.update(pca_refit_sha256=sha(dest),numerical_gate_pass=passed,serialization_exact=serialized_exact,
                 status='PCA_REFIT_NUMERICAL_PARITY_PASS' if passed and serialized_exact else 'PCA_REFIT_DIFFERS_FROM_FROZEN_RETAIN_ARCHIVE')
    require(all(sha(p)==h for p,h in paths.items()),'INPUT_CHANGED');r['all_input_pins_unchanged']=True


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('self-test')
    for mode in ('validate','fit'):
        s=sub.add_parser(mode);s.add_argument('--matrices-dir',required=True);s.add_argument('--teacher-export',required=True);s.add_argument('--frozen-pca',required=True);s.add_argument('--out-dir',required=True)
    a=p.parse_args();core=companion()
    if a.command=='self-test':self_test(core);return
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False)
    r=dict(version='1.1',status='BLOCKED',command=a.command,pca_fitted=False,prediction_model_training=False,
           official_TEST_or_VAL_read=False,threshold_changes=False,original_assets_replaced=False,
           release_authorization_claimed=False,numpy_version=np.__version__,pandas_version=pd.__version__,sklearn_version=sklearn.__version__)
    try:run(a,out,r,core)
    except Exception as e:
        # Do not export paths, row values or full exception messages.
        r.update(status='BLOCKED',exception_type=type(e).__name__,check_code=str(e) if type(e) is ValueError and str(e) in {'CASE_OVERLAP','EMBARGO','OOS_CENSUS','FROZEN_PCA_CONTRACT','FROZEN_TRANSFORM_PARITY','PCA_FIT_INPUT','PCA_FIT_OUTPUT','EMBEDDING_ALIGNMENT','EMBEDDING_VALUES','COMPARE','INPUT_CHANGED'} else 'LOCAL_REVIEW_REQUIRED')
    (out/'summary.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf8');print(r['status'])
    if r['status'] in ('BLOCKED','PCA_REFIT_DIFFERS_FROM_FROZEN_RETAIN_ARCHIVE'):raise SystemExit(1)


if __name__=='__main__':main()
