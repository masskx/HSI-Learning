"""Reproducible, small lecture runs. Historical results are never overwritten.

All preprocessors fit training centers only. Spatial patches still share scene
context; class-disjoint labels do NOT imply spatially disjoint observations.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .data import load_hsi_dataset, split_ground_truth
from .teaching import (EpisodeSampler, PatchClassifier, PatchEncoder, calibrate_threshold,
                       fit_preprocessing, patches_at, prototype_logits, prototypes,
                       rejection_predictions, transform_cube, unknown_scores)
from .utils import save_json, set_seed

ROOT = Path(__file__).resolve().parents[2]
META_CLASSES = {'train': [2, 3, 5, 6, 10, 11],
                'val': [1, 4, 7, 8, 9], 'test': [12, 13, 14, 15, 16]}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    paths = [ROOT/'src/hsi_learning/teaching.py', Path(__file__), ROOT/'src/hsi_learning/data.py']
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unavailable'
    return dict(git_commit=commit, source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in paths},
                python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                data_sha256={p.name: sha256(p) for p in (ROOT/'dataset').glob('Indian_pines*.mat')})


@torch.no_grad()
def encode_ids(model, cube, ids, patch_size=9, batch_size=128, classifier=False):
    model.eval()
    chunks = []
    for start in range(0, len(ids), batch_size):
        x = patches_at(cube, ids[start:start+batch_size], patch_size)
        chunks.append((model.encoder(x) if classifier else model(x)).cpu())
    return torch.cat(chunks)


def save_bundle(out, model, prep, split, config, history, metrics):
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out/'model.pth')
    np.savez(out/'preprocessing.npz', **prep)
    np.savez(out/'split.npz', **split)
    save_json(config, out/'run_config.json')
    save_json(history, out/'history.json')
    save_json(metrics, out/'metrics.json')
    save_json({p.name: sha256(p) for p in out.iterdir() if p.is_file() and p.name != 'hashes.json'},
              out/'hashes.json')


def load_bundle(out):
    out = Path(out)
    if not (out/'hashes.json').exists():
        raise FileNotFoundError(f'Missing verified demo bundle: {out}. Run scripts/prepare_teaching_artifacts.py first.')
    hashes = json.loads((out/'hashes.json').read_text(encoding='utf-8'))
    required = {'model.pth', 'preprocessing.npz', 'split.npz', 'run_config.json',
                'history.json', 'metrics.json'}
    if not required <= hashes.keys():
        raise ValueError(f'Incomplete bundle manifest: missing {sorted(required - hashes.keys())}')
    for name, expected in hashes.items():
        if Path(name).name != name or name in {'.', '..'}:
            raise ValueError(f'Invalid artifact name: {name}')
        if sha256(out/name) != expected:
            raise ValueError(f'Artifact hash mismatch: {name}')
    cfg = json.loads((out/'run_config.json').read_text(encoding='utf-8'))
    model = PatchEncoder(cfg['components']) if cfg['task'] == 'protonet' else PatchClassifier(cfg['components'], len(cfg['classes']))
    model.load_state_dict(torch.load(out/'model.pth', map_location='cpu', weights_only=True))
    model.eval()
    prep = dict(np.load(out/'preprocessing.npz', allow_pickle=False))
    split = dict(np.load(out/'split.npz', allow_pickle=False))
    cube, gt, names = load_hsi_dataset('IP', ROOT/'dataset')
    for filename, expected in cfg['provenance']['data_sha256'].items():
        if sha256(ROOT/'dataset'/filename) != expected:
            raise ValueError(f'Dataset differs from demo manifest: {filename}')
    return model, transform_cube(cube, prep), gt, names, split, cfg


def episode_forward(model, cube, ep, patch_size, training=False):
    sx = patches_at(cube, ep.support_ids, patch_size)
    qx = patches_at(cube, ep.query_ids, patch_size)
    # Fixed BN inference statistics in episodic training avoid query-batch adaptation.
    model.eval()
    se, qe = model(sx), model(qx)
    logits = prototype_logits(se, torch.as_tensor(ep.support_y), qe, len(ep.classes))
    return logits, torch.as_tensor(ep.query_y)


def evaluate_tasks(model, cube, tasks, patch_size):
    accs = []
    with torch.no_grad():
        for ep in tasks:
            logits, target = episode_forward(model, cube, ep, patch_size)
            accs.append(float((logits.argmax(1) == target).float().mean()))
    return accs


def run_protonet(args):
    set_seed(args.seed)
    raw, gt, _ = load_hsi_dataset('IP', ROOT/'dataset')
    labels = gt.ravel()
    if args.mode == 'class-disjoint':
        classes = META_CLASSES
        pools = {k: np.flatnonzero(np.isin(labels, v)) for k, v in classes.items()}
    else:
        maps = split_ground_truth(gt, .1, .1, random_state=args.seed)
        pools = {k: np.flatnonzero(m) for k, m in zip(['train','val','test'], maps)}
        # Fix the eligible class pool across all K rather than changing task difficulty silently.
        eligible = [c for c in range(1, 17) if all(np.sum(labels[p] == c) >= 15 for p in pools.values())]
        classes = {k: eligible for k in pools}
    cube, prep = fit_preprocessing(raw, pools['train'], args.components)
    samplers = {k: EpisodeSampler(labels, pools[k], classes[k], args.seed+offset, 15)
                for k, offset in [('train',0),('val',1000),('test',2000)]}
    val_tasks = [samplers['val'].sample(5, 5, 5) for _ in range(args.eval_episodes)]
    encoder = PatchEncoder(args.components)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    history = {'episode': [], 'loss': [], 'val_accuracy': []}
    best, best_state, best_step = -1., None, 0
    for step in range(1, args.episodes+1):
        ep = samplers['train'].sample(5, 5, 5)
        logits, target = episode_forward(encoder, cube, ep, args.patch_size, True)
        optimizer.zero_grad()
        loss = F.cross_entropy(logits, target); loss.backward(); optimizer.step()
        history['loss'].append(float(loss.detach()))
        if step == 1 or step % 50 == 0 or step == args.episodes:
            score = float(np.mean(evaluate_tasks(encoder, cube, val_tasks, args.patch_size)))
            history['episode'].append(step); history['val_accuracy'].append(score)
            if score > best:
                best, best_state, best_step = score, copy.deepcopy(encoder.state_dict()), step
            print(f'episode={step} val={score:.4f}', flush=True)
    encoder.load_state_dict(best_state); encoder.eval()
    # Same class selections and pixel draws across K: first 10 for support, last 5 for query.
    master = [samplers['test'].sample(5, 10, 5) for _ in range(args.eval_episodes)]
    from .teaching import Episode
    scores = {}
    for shot in [1, 5, 10]:
        tasks = [Episode(e.classes, e.support_ids.reshape(5,10)[:,:shot].ravel(), e.query_ids,
                         np.repeat(np.arange(5),shot), e.query_y) for e in master]
        values = evaluate_tasks(encoder, cube, tasks, args.patch_size)
        scores[str(shot)] = {'mean': float(np.mean(values)), 'episode_std': float(np.std(values, ddof=1)), 'accuracies': values}
    config = dict(task='protonet', mode=args.mode, components=args.components, patch_size=args.patch_size,
                  seed=args.seed, classes=classes, episodes=args.episodes, eval_episodes=args.eval_episodes,
                  best_step=best_step, bn='eval during episodes; no query adaptation',
                  preprocessing='PCA+scaler fit meta-training centers only', provenance=provenance())
    split = {**{f'{k}_ids': v for k,v in pools.items()},
             'episode_classes': np.stack([e.classes for e in master]),
             'episode_support': np.stack([e.support_ids for e in master]),
             'episode_query': np.stack([e.query_ids for e in master])}
    save_bundle(Path(args.output_dir), encoder, prep, split, config, history,
                dict(kshot=scores, best_val=best, annotation_budget=len(pools['train'])))


def run_classifier(args):
    set_seed(args.seed)
    raw, gt, _ = load_hsi_dataset('IP', ROOT/'dataset'); labels = gt.ravel()
    classes = [c for c in range(1,17) if args.task != 'openset' or c not in args.unknown_classes]
    if args.task == 'openset' and (not args.unknown_classes or len(classes) < 2):
        raise ValueError('Need known and unknown classes')
    known_gt = np.where(np.isin(gt, classes), gt, 0)
    train, val, test = split_ground_truth(known_gt, .1, .1, random_state=args.seed)
    pools = dict(train_ids=np.flatnonzero(train), val_ids=np.flatnonzero(val),
                 test_ids=np.concatenate([np.flatnonzero(test), np.flatnonzero(np.isin(labels,args.unknown_classes))])
                 if args.task == 'openset' else np.flatnonzero(test))
    cube, prep = fit_preprocessing(raw, pools['train_ids'], args.components)
    mapping = {c:i for i,c in enumerate(classes)}
    yt = torch.tensor([mapping[c] for c in labels[pools['train_ids']]])
    yv = torch.tensor([mapping[c] for c in labels[pools['val_ids']]])
    train_x = patches_at(cube, pools['train_ids'], args.patch_size)
    val_x = patches_at(cube, pools['val_ids'], args.patch_size)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(train_x, yt), batch_size=64, shuffle=True, generator=generator)
    model = PatchClassifier(args.components,len(classes))
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    counts = torch.bincount(yt,minlength=len(classes)).float()
    weight = len(yt)/(len(classes)*counts.clamp_min(1)) if args.loss == 'weighted' else None
    history = dict(train_loss=[], train_accuracy=[], val_accuracy=[])
    best, state, best_epoch = -1., None, 0
    for epoch in range(1,args.epochs+1):
        model.train(); total_loss=correct=0
        for x,y in loader:
            optimizer.zero_grad(); logits=model(x)
            if args.loss == 'focal':
                ce = F.cross_entropy(logits, y, reduction='none')
                loss = (((1-torch.exp(-ce))**2)*ce).mean()
            else:
                # Match CrossEntropyLoss(weight=...): normalize by batch target weights.
                loss = F.cross_entropy(logits, y, weight=weight)
            loss.backward(); optimizer.step()
            total_loss+=float(loss.detach())*len(y); correct+=int((logits.argmax(1)==y).sum())
        model.eval()
        with torch.no_grad():
            vl=torch.cat([model(v) for v in val_x.split(128)])
            acc=float((vl.argmax(1)==yv).float().mean())
        history['train_loss'].append(total_loss/len(yt)); history['train_accuracy'].append(correct/len(yt)); history['val_accuracy'].append(acc)
        if acc > best: best,state,best_epoch=acc,copy.deepcopy(model.state_dict()),epoch
        if epoch%5==0 or epoch==1: print(f'epoch={epoch} val={acc:.4f}',flush=True)
    model.load_state_dict(state); model.eval()
    train_emb=encode_ids(model,cube,pools['train_ids'],args.patch_size,classifier=True)
    centers=prototypes(train_emb,yt,range(len(classes)))
    config=dict(task=args.task,components=args.components,patch_size=args.patch_size,seed=args.seed,
                classes=classes,unknown_classes=args.unknown_classes if args.task=='openset' else [],
                loss=args.loss,epochs=args.epochs,best_epoch=best_epoch,
                preprocessing='PCA+scaler fit known training centers only',provenance=provenance())
    with torch.no_grad():
        ve=encode_ids(model,cube,pools['val_ids'],args.patch_size,classifier=True)
        vs=unknown_scores(model.head(ve),ve,centers)
        thresholds={k:calibrate_threshold(s.numpy(),.95) for k,s in zip(['msp','distance'],vs)}
        te=encode_ids(model,cube,pools['test_ids'],args.patch_size,classifier=True)
        logits=model.head(te); scores=unknown_scores(logits,te,centers)
    true=labels[pools['test_ids']]; known=np.isin(true,classes); unknown_id=17
    pred=np.asarray(classes)[logits.argmax(1).numpy()]
    metrics=dict(known_accuracy=float(accuracy_score(true[known],pred[known])),best_val=best)
    arrays=dict(test_ids=pools['test_ids'],true=true,predicted=pred,prototypes=centers.numpy())
    for method, score in zip(['msp','distance'],scores):
        score=score.numpy(); rejected=rejection_predictions(pred,score,thresholds[method],unknown_id)
        arrays[method+'_score']=score; arrays[method+'_prediction']=rejected
        if args.task=='openset':
            metrics[method]=dict(auroc=float(roc_auc_score(~known,score)),
                known_false_rejection=float(np.mean(rejected[known]==unknown_id)),
                known_correct_acceptance=float(np.mean(rejected[known]==true[known])),
                unknown_recall=float(np.mean(rejected[~known]==unknown_id)),
                unknown_miss=float(np.mean(rejected[~known]!=unknown_id)))
    config['thresholds']=thresholds; config['acceptance']=.95; config['unknown_id']=unknown_id
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    np.savez(out/'predictions.npz',**arrays)
    save_bundle(out,model,prep,pools,config,history,metrics)


def main(task=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['protonet','openset','classifier'],default=task or 'openset')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--components',type=int,default=12)
    parser.add_argument('--patch-size',type=int,default=9)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--episodes',type=int,default=200)
    parser.add_argument('--eval-episodes',type=int,default=30)
    parser.add_argument('--mode',choices=['class-disjoint','same-class'],default='class-disjoint')
    parser.add_argument('--unknown-classes',type=int,nargs='+',default=[1,7,9,16])
    parser.add_argument('--loss',choices=['ce','weighted','focal'],default='ce')
    args=parser.parse_args()
    if min(args.epochs,args.episodes,args.eval_episodes)<2: parser.error('budgets must be >=2')
    if len(set(args.unknown_classes))!=len(args.unknown_classes) or not set(args.unknown_classes)<=set(range(1,17)):
        parser.error('unknown class IDs must be distinct and in 1..16')
    args.output_dir=args.output_dir or ROOT/'results/teaching_ready'/args.task
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f'Output is not empty; choose a new run directory: {args.output_dir}')
    torch.set_num_threads(2)
    if args.task=='protonet': run_protonet(args)
    else: run_classifier(args)
    print(f'Saved verified run inputs and outputs to {args.output_dir}',flush=True)


if __name__=='__main__':
    main()
