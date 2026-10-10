"""Classroom experiments with explicit sample and metric boundaries."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from .data import split_ground_truth
from .teaching import calibrate_threshold, rejection_predictions

def classroom_split(gt, seed=42):
    """Protocol E: shared 10/10/80 centers, identical to new lecture runs."""
    maps = split_ground_truth(gt, .1, .1, random_state=seed)
    return {key: np.flatnonzero(array) for key, array in zip(('train_ids','val_ids','test_ids'), maps)}

def classification_summary(true, predicted, class_ids):
    """Rows follow caller IDs. Absent test classes have undefined recall."""
    ids = np.asarray(class_ids)
    if not len(true) or not len(ids):
        raise ValueError('Need evaluation samples and declared class IDs')
    if not np.isin(true,ids).all() or not np.isin(predicted,ids).all():
        raise ValueError('Truth and predictions must use declared class IDs')
    matrix = confusion_matrix(true, predicted, labels=ids)
    support = matrix.sum(1)
    recall = np.divide(np.diag(matrix), support, out=np.full(len(ids), np.nan), where=support>0)
    return dict(oa=float(accuracy_score(true,predicted)), aa=float(np.nanmean(recall)),
                kappa=float(cohen_kappa_score(true,predicted)), class_ids=ids.tolist(),
                recall=[None if np.isnan(v) else float(v) for v in recall],
                support=support.tolist(), confusion_matrix=matrix.tolist())

def svm_pipeline(c=10, components=None):
    steps = [] if components is None else [PCA(n_components=components,whiten=True,svd_solver='full')]
    return make_pipeline(*steps,StandardScaler(),SVC(C=c,gamma='scale',cache_size=512))

def fit_svm_baseline(cube,gt,split,components=None,candidates=(1,10,100)):
    x,y=cube.reshape(-1,cube.shape[-1]),gt.ravel()
    train,val,test=(split[k] for k in ('train_ids','val_ids','test_ids'))
    if any(set(a)&set(b) for a,b in ((train,val),(train,test),(val,test))):
        raise ValueError('Train, validation and test centers must be disjoint')
    selection=[]; best=-1.; model=None
    for c in candidates:
        candidate=svm_pipeline(c,components).fit(x[train],y[train])
        score=float(candidate.score(x[val],y[val]))
        selection.append(dict(c=c,validation_oa=score))
        if score>best: best,model=score,candidate
    pred=model.predict(x[test])
    return model,classification_summary(y[test],pred,np.unique(y[y>0])),selection,pred

def threshold_experiment(true,predicted,scores,known_classes,validation_scores,acceptance=.95,unknown_id=17):
    """New val-only thresholds and freshly evaluated metrics; no test tuning."""
    true,predicted=np.asarray(true),np.asarray(predicted)
    known=np.isin(true,known_classes)
    if not known.any() or known.all(): raise ValueError('Need known and unknown evaluation samples')
    thresholds,metrics,outputs={},{},{}
    for method,score in scores.items():
        tau=calibrate_threshold(validation_scores[method],acceptance)
        reject=rejection_predictions(predicted,score,tau,unknown_id)
        thresholds[method],outputs[method]=tau,reject
        metrics[method]=dict(auroc=float(roc_auc_score(~known,score)),
            known_false_rejection=float(np.mean(reject[known]==unknown_id)),
            known_correct_acceptance=float(np.mean(reject[known]==true[known])),
            known_wrong_acceptance=float(np.mean((reject[known]!=unknown_id)&(reject[known]!=true[known]))),
            unknown_recall=float(np.mean(reject[~known]==unknown_id)),
            unknown_miss=float(np.mean(reject[~known]!=unknown_id)),
            known_n=int(known.sum()),unknown_n=int((~known).sum()))
    return thresholds,metrics,outputs

def spatial_split(gt,patch_size=9,fractions=(.4,.2),axis=1):
    """Contiguous regions with radius margins on BOTH sides of boundaries.

    Changes task and class coverage; cannot estimate a pure leakage benefit.
    """
    if patch_size<3 or patch_size%2!=1 or axis not in (0,1): raise ValueError('Odd size >=3, axis 0/1 required')
    a,b=fractions
    if min(a,b)<=0 or a+b>=1: raise ValueError('Invalid train/val fractions')
    length,radius=gt.shape[axis],patch_size//2
    first,second=int(length*a),int(length*(a+b))
    coordinates=np.indices(gt.shape)[axis]
    regions=[coordinates<first-radius,(coordinates>=first+radius)&(coordinates<second-radius),coordinates>=second+radius]
    return {key:np.flatnonzero(region&(gt>0)) for key,region in zip(('train_ids','val_ids','test_ids'),regions)}

def overlap_fraction(offset,size=9):
    """Pairwise sharing; not an estimate of whole-dataset leakage."""
    dr,dc=map(abs,offset)
    return max(0,size-dr)*max(0,size-dc)/(size*size)
