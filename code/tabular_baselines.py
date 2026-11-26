#!/usr/bin/env python3
"""Five tabular comparator recipes on prepared research TRAIN matrices.

Requires primary_student.py alongside this module. Fit is optional and creates
new local artifacts only; reported frozen models are not modified. Published
thresholds are retained as constants, not selected anew. Exact archived refits
are not claimed. Model files produced by a local fit require disclosure review
before redistribution. No teacher/PCA fit or held-out access.
"""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from catboost import CatBoostClassifier

COMPANION_SHA = 'a411d6dc5608efc3fd776c75bd239b1dd32b60f933d18e78fd6f756afd24a2f6'
companion = Path(__file__).resolve().with_name('primary_student.py')
if not companion.is_file() or hashlib.sha256(companion.read_bytes()).hexdigest() != COMPANION_SHA:
    raise RuntimeError('Place the verified primary_student.py alongside this script.')
import primary_student as core
if Path(core.__file__).resolve() != companion:
    raise RuntimeError('Unexpected companion import location.')

ARMS = {
    'observed_only_hard': core.BASE,
    'catboost_observed19_hard': core.BASE,
    'logit_hard': core.BASE + ['teacher_logit'],
    'pca48_hard': core.BASE + core.PC,
    'full_hard': core.COLS,
}
THRESHOLDS = {
    'observed_only_hard': (0.36, 0.77),
    'catboost_observed19_hard': (0.30, 0.76),
    'logit_hard': (0.31, 0.38),
    'pca48_hard': (0.27, 0.44),
    'full_hard': (0.28, 0.47),
}
CAT_CONFIG = dict(iterations=400, depth=6, learning_rate=0.05,
    l2_leaf_reg=3.0, loss_function='Logloss', eval_metric='Logloss',
    random_seed=42, thread_count=4, verbose=False, allow_writing_files=False,
    bootstrap_type='Bernoulli', subsample=0.8, rsm=0.9,
    border_count=128, nan_mode='Forbidden')


def projections(parts):
    result = {}
    for arm, cols in ARMS.items():
        core.need(len(cols) == len(set(cols)), 'FEATURE_DUPLICATES')
        result[arm] = {}
        for stage, (x, s, m) in parts.items():
            z = x[cols].astype(np.float32)
            core.need(list(z) == cols and len(z) == len(s) and
                      np.isfinite(z.to_numpy()).all(), 'PROJECTION')
            core.need(s.y_next_delay.isin([0, 1]).all(), 'HARD_LABELS')
            if stage != 'threshold_selection':
                core.need(s.is_predeadline.astype(bool).all() and
                          s.weight_predeadline.gt(0).all() and
                          np.isfinite(s.weight_predeadline).all(), 'WEIGHTS')
            result[arm][stage] = z
    return result


def fit_one(arm, projected, parts, out, synthetic=False):
    xf, xc = projected['fit'], projected['calibration']
    fs, cs = parts['fit'][1], parts['calibration'][1]
    yf, yc = fs.y_next_delay.to_numpy(np.int8), cs.y_next_delay.to_numpy(np.int8)
    wf, wc = fs.weight_predeadline.to_numpy(np.float64), cs.weight_predeadline.to_numpy(np.float64)
    core.need(set(yf) == {0,1} and set(yc) == {0,1}, 'CLASS_SUPPORT')
    out.mkdir(exist_ok=False)
    if arm == 'catboost_observed19_hard':
        cfg = dict(CAT_CONFIG)
        if synthetic: cfg['iterations'] = 12
        model = CatBoostClassifier(**cfg)
        model.fit(xf, yf, sample_weight=wf)
        predict = lambda x: model.predict_proba(x)[:,1].astype(np.float64)
        path = out/'model.cbm'
        model.save_model(str(path))
        reloaded = CatBoostClassifier()
        reloaded.load_model(str(path))
        reload_predict = lambda x: reloaded.predict_proba(x)[:,1].astype(np.float64)
    else:
        model = core.fit_student(xf,yf,wf,12 if synthetic else core.N_TREES)
        predict = lambda x: core.predict(model,x)
        path = out/'model.json'
        model.save_model(str(path))
        reloaded = xgb.Booster(); reloaded.load_model(str(path))
        reload_predict = lambda x: core.predict(reloaded,x)
    cal = core.calibrate(predict(xc),yc,wc)
    cp = out/'calibrator.joblib'
    joblib.dump(cal,cp,compress=3)
    sample = xc.iloc[:23]
    before = core.calibrated(cal,predict(sample))
    after = core.calibrated(joblib.load(cp),reload_predict(sample))
    error = float(np.max(np.abs(before-after)))
    core.need(error <= 1e-7,'SERIALIZATION')
    return {'asset_sha256':{path.name:core.sha(path),cp.name:core.sha(cp)},
            'calibrated_serialization_max_abs':error,
            'archived_fit_equivalence_claimed':False}


def self_test():
    rng = np.random.default_rng(42)
    parts = {}
    for stage in ('fit','calibration','threshold_selection'):
        x = pd.DataFrame(rng.normal(size=(180,68)).astype(np.float32),columns=core.COLS)
        s = pd.DataFrame({'y_next_delay':(x.iloc[:,0]>0).astype(np.int8),
                          'weight_predeadline':np.ones(len(x)), 'is_predeadline':True})
        parts[stage] = (x,s,None)
    p = projections(parts)
    core.need([len(c) for c in ARMS.values()]==[19,19,20,67,68],'DIMENSIONS')
    with tempfile.TemporaryDirectory() as t:
        for arm in ARMS:
            fit_one(arm,p[arm],parts,Path(t)/arm,synthetic=True)
    print('Five synthetic fits, calibration, feature projection and serialization checks passed.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    subs.add_parser('self-test')
    for cmd in ('validate','fit'):
        p=subs.add_parser(cmd)
        p.add_argument('--matrices-dir',required=True)
        p.add_argument('--out-dir',required=True)
        if cmd=='fit': p.add_argument('--arm',choices=list(ARMS)+['all'],required=True)
    args=parser.parse_args()
    if args.command=='self-test': self_test(); return
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=False)
    r=dict(version='1.0',status='BLOCKED',command=args.command,
           training_performed=False,official_TEST_or_VAL_read=False,
           threshold_search=False,teacher_or_pca_refit=False,
           original_assets_replaced=False,archived_model_equivalence_claimed=False,
           release_authorization_claimed=False)
    try:
        paths={}
        parts={k:core.read_stage(Path(args.matrices_dir),k,paths) for k in core.COUNTS}
        for a,b in [('fit','calibration'),('fit','threshold_selection'),('calibration','threshold_selection')]:
            core.need(set(parts[a][0].case_seg).isdisjoint(parts[b][0].case_seg),'CASE_OVERLAP')
            core.need(pd.to_datetime(parts[a][2].t_next,utc=True).max()+pd.Timedelta(hours=2)<pd.to_datetime(parts[b][2].t0,utc=True).min(),'EMBARGO')
        projected=projections(parts)
        r['arms']={arm:{'feature_order':cols,'feature_count':len(cols),
                        'stage_rows':{k:len(z) for k,z in projected[arm].items()},
                        'primary_fixed_threshold':THRESHOLDS[arm][0],
                        'D5_fixed_threshold':THRESHOLDS[arm][1],
                        'supervision':'hard binary labels',
                        'projection_validated':True} for arm,cols in ARMS.items()}
        if args.command=='fit':
            for arm in (list(ARMS) if args.arm=='all' else [args.arm]):
                r['training_performed']=True
                r['arms'][arm]['refit']=fit_one(arm,projected[arm],parts,out/arm)
        r['all_input_pins_unchanged']=all(core.sha(p)==h for p,h in paths.items())
        core.need(r['all_input_pins_unchanged'],'INPUT_CHANGED')
        r['status']='FIVE_TABULAR_INPUT_PROJECTIONS_PASS' if args.command=='validate' else 'LOCAL_REFIT_COMPLETE_NOT_ARCHIVED_MODEL'
    except Exception as e:
        r.update(status='BLOCKED',exception_type=type(e).__name__,
                 check_code=str(e) if isinstance(e,core.GateError) else 'LOCAL_REVIEW_REQUIRED')
    (out/'summary.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(r['status'])
    if r['status']=='BLOCKED': raise SystemExit(1)

if __name__=='__main__': main()
