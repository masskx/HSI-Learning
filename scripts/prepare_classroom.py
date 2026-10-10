"""Prepare val-selected SVMs with the shared classroom center split."""
from pathlib import Path
import argparse,hashlib,sys,time,platform
import joblib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hsi_learning.data import load_hsi_dataset
from hsi_learning.classroom import classroom_split,fit_svm_baseline
from hsi_learning.utils import save_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/classroom/svm')
    args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    cube,gt,names=load_hsi_dataset('IP',ROOT/'dataset'); split=classroom_split(gt)
    np.savez(args.output_dir/'split.npz',**split); outcomes={}
    for variant,components in [('raw',None),('pca12',12)]:
        started=time.perf_counter()
        model,metrics,selection,predicted=fit_svm_baseline(cube,gt,split,components)
        joblib.dump(model,args.output_dir/f'{variant}.joblib')
        full=model.predict(cube.reshape(-1,cube.shape[-1])).reshape(gt.shape)
        np.savez(args.output_dir/f'{variant}-predictions.npz',test=predicted,full=full)
        outcomes[variant]=dict(metrics=metrics,selection=selection,elapsed_seconds=time.perf_counter()-started)
        print(variant,metrics['oa'],metrics['aa'],flush=True)
    config=dict(protocol='E',seed=42,rates=[.1,.1,.8],preprocessing_fit='training centers only',
        environment={'python':platform.python_version(),'numpy':np.__version__,
                     'sklearn':__import__('sklearn').__version__},
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [ROOT/'src/hsi_learning/classroom.py',ROOT/'src/hsi_learning/data.py']},
        selection='highest validation OA; ties keep first C',candidates=[1,10,100],class_names=names,
        data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'dataset').glob('Indian_pines*.mat')})
    save_json(config,args.output_dir/'run_config.json'); save_json(outcomes,args.output_dir/'metrics.json')
    files=['run_config.json','metrics.json','split.npz','raw.joblib','pca12.joblib',
           'raw-predictions.npz','pca12-predictions.npz']
    save_json({name:hashlib.sha256((args.output_dir/name).read_bytes()).hexdigest() for name in files},args.output_dir/'hashes.json')
if __name__=='__main__': main()
