"""Build lecture notebooks with stable cell IDs and atomic execution.

Existing notebooks are replaced only after successful execution. Use
--draft-only to inspect source without claiming embedded outputs are verified.
Kernel is explicitly bound to this Python, not a user's unrelated python3.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from lecture_notebook_validation import output_issues

ROOT=Path(__file__).resolve().parents[1]
BOOT='''from pathlib import Path
import sys
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'src/hsi_learning').is_dir() and (p / 'dataset').is_dir()), None)
if ROOT is None:
    raise RuntimeError('Open this notebook from inside the HSI-Learning repository')
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
# Notebook kernels must explicitly use inline rendering even when the CLI
# parent uses MPLBACKEND=Agg for non-interactive training scripts.
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('png')
import matplotlib.pyplot as plt
from hsi_learning.data import load_hsi_dataset
from hsi_learning.teaching_runs import load_bundle, encode_ids
from hsi_learning.teaching import patches_at, prototype_logits, prototypes, grad_cam
from hsi_learning.teaching_plots import style, maps
style()
torch.set_num_threads(2)
print('Python:', sys.executable)
print('Repository:', ROOT)
'''
MODE='''# Classroom mode NEVER silently starts training.
MODE = 'demo'  # 'train' explicitly starts the full preparation queue below
BUNDLE_ROOT = ROOT / 'results/teaching_ready'
if MODE == 'train':
    import subprocess
    subprocess.run([sys.executable, str(ROOT/'scripts/prepare_teaching_artifacts.py')], check=True)
elif MODE != 'demo':
    raise ValueError('MODE must be demo or train')
'''


def md(id,text):
    c=nbformat.v4.new_markdown_cell(text); c.id=id; return c


def code(id,text):
    c=nbformat.v4.new_code_cell(text); c.id=id; return c


def common(title,intro):
    return [md('overview',f'# {title}\n\n{intro}\n\n默认 demo 加载预先训练权重，核心算法在下方逐步演示。旧实验数字不作为本次结果。'),
            code('environment',BOOT),code('mode',MODE)]


def environment():
    return [md('overview','# 第 0 课：环境与第一次前向\n\n不训练模型。先验证 kernel、路径、数据与张量接口。'),
            code('environment',BOOT),
            code('dataset',"""import platform, scipy, sklearn
print({'python': platform.python_version(), 'numpy': np.__version__,
       'torch': torch.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__})
cube, gt, names = load_hsi_dataset('IP', ROOT/'dataset')
assert cube.shape[:2] == gt.shape
assert set(np.unique(gt)) <= set(range(len(names)+1))
print('cube:', cube.shape, cube.dtype, 'GT:', gt.shape, 'labeled:', np.count_nonzero(gt))
plt.imshow(gt, cmap='nipy_spectral', vmin=0, vmax=len(names), interpolation='nearest')
plt.title('Indian Pines ground truth (0 = unlabeled)'); plt.axis('off'); plt.show()
"""),
            code('forward',"""from hsi_learning.teaching import PatchClassifier
model = PatchClassifier(12, len(names)).eval()
BATCH_SIZE = 2  # exercise: change this and the input together
with torch.no_grad(): logits = model(torch.zeros(BATCH_SIZE,12,9,9))
assert logits.shape == (BATCH_SIZE,len(names))
print('Input', (BATCH_SIZE,12,9,9), '-> logits', tuple(logits.shape))
print('Random weights: this is an interface check, NOT a classification result.')
"""),md('exercise','## 练习\n把 batch 改成3、类别数改成9；预测输出形状并验证。排错先检查 sys.executable，不要盲目重装包。')]


def fewshot():
    cells=common('原型网络：5-shot 的五个样本是什么？','分清 meta-training 标签预算与单任务支持集。主示范采用类别不相交划分；同场景 patch 仍可能共享空间上下文。')
    cells += [code('protocol',"""model, cube, gt, names, split, cfg = load_bundle(BUNDLE_ROOT/'protonet')
print('Class split:', cfg['classes'])
print('Meta-training annotated centers:', len(split['train_ids']))
print('Preprocessing:', cfg['preprocessing'])
assert not (set(cfg['classes']['train']) & set(cfg['classes']['test']))
"""),md('episode-explanation','## Support / Query\n每类10个可用支持位置、5个查询位置。缩小 K 时保留相同查询任务；不得把查询标签用于建原型。'),
code('episode',"""from hsi_learning.teaching import Episode
from hsi_learning.teaching_plots import episode_figure
classes = split['episode_classes'][0]
support = split['episode_support'][0].reshape(5,10)[:,:5].ravel()
query = split['episode_query'][0]
ep = Episode(classes, support, query, np.repeat(np.arange(5),5), np.repeat(np.arange(5),5))
assert not set(support) & set(query)
print('Episode classes:',classes,'support:',len(support),'query:',len(query))
episode_figure(gt,ep,names)
"""),md('math-explanation','## 类均值与负平方距离\n先手算二维例子，再对真实嵌入运行同一函数。'),
code('prototype-math',"""s=torch.tensor([[0.,0.],[2.,0.],[8.,0.],[10.,0.]])
y=torch.tensor([0,0,1,1]); q=torch.tensor([[4.,0.]])
print('Hand-check logits (expected -9,-25):',prototype_logits(s,y,q,2))
with torch.no_grad():
    se=model(patches_at(cube,support)); qe=model(patches_at(cube,query))
    logits=prototype_logits(se,torch.as_tensor(ep.support_y),qe,5)
print('Episode accuracy:', float((logits.argmax(1)==torch.as_tensor(ep.query_y)).float().mean()))
"""),
code('episode-update',"""from copy import deepcopy
from hsi_learning.teaching import EpisodeSampler
from hsi_learning.teaching_runs import episode_forward
# Update only a COPY and only on meta-training classes, never test support/query.
train_sampler=EpisodeSampler(gt.ravel(),split['train_ids'],cfg['classes']['train'],42,15)
train_ep=train_sampler.sample(5,5,5)
learner=deepcopy(model); optimizer=torch.optim.Adam(learner.parameters(),lr=1e-4)
before=next(learner.parameters()).detach().clone()
train_logits,train_targets=episode_forward(learner,cube,train_ep,cfg['patch_size'],True)
optimizer.zero_grad(); loss=torch.nn.functional.cross_entropy(train_logits,train_targets)
loss.backward(); optimizer.step()
print('Support -> prototypes -> query logits:',train_logits.shape,'CE:',float(loss.detach()))
print('Meta-train gradient update:',float((next(learner.parameters()).detach()-before).norm()))
torch.testing.assert_close(next(model.parameters()),before)
"""),code('evaluation',"""import json
metrics=json.loads((BUNDLE_ROOT/'protonet/metrics.json').read_text())
print('Same encoder; fixed class pool and paired queries:')
for k,v in metrics['kshot'].items(): print(k, 'shot:', v['mean'], 'episode std:',v['episode_std'])
plt.plot([1,5,10],[metrics['kshot'][str(k)]['mean'] for k in [1,5,10]],'o-')
plt.xticks([1,5,10]); plt.xlabel('Support samples per class'); plt.ylabel('Episode accuracy')
plt.ylim(0,1); plt.title('Same frozen encoder, paired tasks'); plt.show()
"""),md('map-explanation','## 全图演示（不是另一项测试成绩）\n只用当前 episode 的五类支持集建原型。全图像元会被强制分到这五类；其他类别不是本任务的可识别类别。GT 仅用于显示与诊断。'),
code('maps',"""all_emb=encode_ids(model,cube,np.arange(gt.size))
with torch.no_grad():
    centers=prototypes(se,torch.as_tensor(ep.support_y),range(5))
    nearest=torch.cdist(all_emb,centers).argmin(1).numpy()
pred=classes[nearest].reshape(gt.shape)
panels={'GT (all classes)':gt,'Forced 5-class prediction':np.where(gt>0,pred,0)}
maps(gt,panels,names,'Prototype map; only the five support classes are available')
print('Support IDs and class mapping are recorded in split.npz; this map is not 16-class OA.')
"""),md('exercise','## 练习与限制\n固定查询，只改 K，再检查个别任务是否一定提升。解释为什么25不是encoder总训练标签数。\n\n类别不交不等于空间不交；训练/验证/测试任务在CLI中独立。经典来源：Snell et al., NeurIPS 2017。')]
    return cells


def openset():
    cells=common('开集识别：白色区域一定识别对了吗？','仅已知类监督训练；阈值只由已知验证集校准。背景黑色，未知/拒识白色，错误图必须与预测图一起看。')
    cells += [code('protocol',"""model,cube,gt,names,split,cfg=load_bundle(BUNDLE_ROOT/'openset')
print('Known:',cfg['classes'],'Unknown:',cfg['unknown_classes'])
print('Threshold calibration:',cfg['acceptance'],'known validation acceptance target')
print('Preprocessing:',cfg['preprocessing'])
"""),
md('scores-explanation','## 两种 unknown score\nMSP分数=1−最大softmax；距离分数=到最近训练类原型距离。均为越大越未知。特征与标签必须同序。'),
code('scores',"""from hsi_learning.teaching import unknown_scores,calibrate_threshold,rejection_predictions
train_ids=split['train_ids']; classes=cfg['classes']
train_emb=encode_ids(model,cube,train_ids,classifier=True)
y=torch.tensor([classes.index(int(c)) for c in gt.ravel()[train_ids]])
centers=prototypes(train_emb,y,range(len(classes)))
print('Prototypes:',tuple(centers.shape))
# Shuffle feature-label PAIRS together: the result must not change.
order=torch.randperm(len(y))
torch.testing.assert_close(centers,prototypes(train_emb[order],y[order],range(len(classes))),atol=1e-5,rtol=1e-5)
"""),code('calibration',"""val_emb=encode_ids(model,cube,split['val_ids'],classifier=True)
with torch.no_grad(): val_scores=unknown_scores(model.head(val_emb),val_emb,centers)
thresholds={key:calibrate_threshold(score.numpy(),cfg['acceptance'])
            for key,score in zip(['msp','distance'],val_scores)}
print('Validation-only thresholds:',thresholds)
for key in thresholds: assert np.isclose(thresholds[key],cfg['thresholds'][key],rtol=1e-5)
# Baseline reproduction above. Change ACCEPTANCE in the separate exercise cell.
"""),
code('evaluation',"""import json
from sklearn.metrics import roc_auc_score,roc_curve
recorded=json.loads((BUNDLE_ROOT/'openset/metrics.json').read_text())
print(json.dumps(recorded,indent=2))
test_ids=split['test_ids']; true=gt.ravel()[test_ids]
known=np.isin(true,classes)
te=encode_ids(model,cube,test_ids,classifier=True)
with torch.no_grad():
    logits=model.head(te); scores=unknown_scores(logits,te,centers)
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for i,(key,score) in enumerate(zip(['msp','distance'],scores)):
    score=score.numpy(); auc=roc_auc_score(~known,score)
    print(key,'AUROC:',auc)
    axes[i].hist(score[known],bins=30,density=True,histtype='step',label='Known')
    axes[i].hist(score[~known],bins=30,density=True,histtype='step',linestyle='--',label='Unknown')
    axes[i].axvline(thresholds[key],color='black',linestyle=':',label='Validation threshold')
    axes[i].set_title(key); axes[i].set_ylabel('Density'); axes[i].legend()
    fpr,tpr,_=roc_curve(~known,score); axes[2].plot(fpr,tpr,label=f'{key}: {auc:.3f}')
axes[2].plot([0,1],[0,1],':',color='gray'); axes[2].set_xlabel('Known false positive rate')
axes[2].set_ylabel('Unknown true positive rate'); axes[2].legend(); plt.show()
"""),
code('maps',"""from hsi_learning.teaching_plots import openset_maps,error_panels
panels,outputs=openset_maps(model,cube,gt,split,cfg)
maps(gt,panels,names,'White = unknown / rejected; black = unlabeled background')
error_panels(gt,outputs,split,cfg)
print('White pixels may be correct rejections OR known-class false rejections.')
"""),code('threshold-exercise',"""from copy import deepcopy
from hsi_learning.classroom import threshold_experiment
ACCEPTANCE = .90  # choose using validation only, before inspecting test outcomes
val_dict={k:s.numpy() for k,s in zip(['msp','distance'],val_scores)}
test_dict={k:s.numpy() for k,s in zip(['msp','distance'],scores)}
predicted=np.asarray(classes)[logits.argmax(1).numpy()]
new_tau,new_metrics,_=threshold_experiment(true,predicted,test_dict,classes,val_dict,ACCEPTANCE)
print('New validation thresholds:',new_tau)
print(json.dumps(new_metrics,indent=2))
exercise_cfg=deepcopy(cfg); exercise_cfg.update(thresholds=new_tau,acceptance=ACCEPTANCE)
exercise_panels,exercise_outputs=openset_maps(model,cube,gt,split,exercise_cfg)
maps(gt,exercise_panels,names,'Acceptance target 0.90; metrics use test centers only')
error_panels(gt,exercise_outputs,split,exercise_cfg)
assert cfg['acceptance']==.95  # original run remains unchanged
"""),md('exercise','## 练习\n把已知接受率目标从.95改成.90，只使用验证分数重新校准；解释误拒与漏检的变化。AUROC是排序指标，不是固定阈值下的分类精度。\n\n本例是同场景中心监督设定，patch可能包含未知地物的无标签观测；不宣称空间外推。OpenMax为概念延伸，未伪装成此处的距离法。')]
    return cells


def imbalance():
    cells=common('类别不均衡与真正的 Grad-CAM','比较损失、逐类召回、地图和类别梯度；不事先断言加权一定改进某一区域。')
    cells += [code('protocol',"""ce,cube,gt,names,split,cfg=load_bundle(BUNDLE_ROOT/'classifier-ce')
weighted,cube_w,gt_w,_,split_w,cfg_w=load_bundle(BUNDLE_ROOT/'classifier-weighted')
np.testing.assert_array_equal(split['test_ids'],split_w['test_ids'])
np.testing.assert_allclose(cube,cube_w)
print('Same training/test centers and preprocessing; loss differs.')
"""),
code('weighted-loss-toy',"""import torch.nn.functional as F
# Build logits whose true-class CE values are exactly .2 and 1.0.
losses=torch.tensor([.2,1.]); probabilities=torch.exp(-losses)
toy_logits=torch.stack([probabilities.log(),(1-probabilities).log()],1)
toy_logits[1]=toy_logits[1].flip(0); targets=torch.tensor([0,1])
weights=torch.tensor([1.,4.])
plain=F.cross_entropy(toy_logits,targets)
weighted_loss=F.cross_entropy(toy_logits,targets,weight=weights)
print('Plain / weighted:',float(plain),float(weighted_loss))
torch.testing.assert_close(plain,torch.tensor(.6))
torch.testing.assert_close(weighted_loss,torch.tensor(.84))
"""),code('loss-comparison',"""import json
from sklearn.metrics import accuracy_score,recall_score,confusion_matrix
true=gt.ravel()[split['test_ids']]; results={}
for label,model in [('CE',ce),('Weighted CE',weighted)]:
    emb=encode_ids(model,cube,split['test_ids'],classifier=True)
    with torch.no_grad(): pred=np.asarray(cfg['classes'])[model.head(emb).argmax(1).numpy()]
    rec=recall_score(true,pred,labels=np.arange(1,17),average=None,zero_division=0)
    results[label]=(pred,rec)
    print(label,'OA:',accuracy_score(true,pred),'AA:',rec.mean(),'per-class recall:',rec)
fig,ax=plt.subplots(figsize=(12,4))
for label,(_,rec) in results.items(): ax.plot(np.arange(1,17),rec,'o-',label=label)
ax.set_xticks(np.arange(1,17)); ax.set_xlabel('Class ID'); ax.set_ylabel('Recall'); ax.set_ylim(0,1.05)
ax.legend(); plt.show()
"""),
code('gradcam',"""from hsi_learning.teaching_plots import cam_figure
# Explicit model selection; target gradients, not a mean activation proxy.
model=ce
cams,logits=cam_figure(model,cube,gt,split['test_ids'],cfg['patch_size'],cfg['classes'],names)
print('CAM shape:',tuple(cams.shape),'finite:',bool(torch.isfinite(cams).all()))
"""),
code('target-comparison',"""pixel=split['test_ids'][0]; inputs=patches_at(cube,[pixel],cfg['patch_size'])
first=int(ce(inputs).argmax(1)); second=(first+1)%len(cfg['classes'])
fig,axes=plt.subplots(1,2,figsize=(7,3),layout='constrained')
for ax,target in zip(axes,[first,second]):
    response,_=grad_cam(ce,ce.encoder.features[7],inputs,torch.tensor([target]))
    ax.imshow(response[0],vmin=0,vmax=1,cmap='magma'); ax.set_title(f'Class ID {cfg["classes"][target]}'); ax.axis('off')
plt.show()
print('Same input, two targets. A zero CAM remains a valid result, not evidence of no dependence.')
"""),code('maps',"""from hsi_learning.teaching_plots import classifier_maps
pred_ce,err_ce=classifier_maps(ce,cube,gt,split,cfg)
pred_w,err_w=classifier_maps(weighted,cube,gt,split,cfg)
maps(gt,{'GT':gt,'CE':pred_ce,'Weighted CE':pred_w},names,'Same-split classification maps')
from hsi_learning.teaching_plots import binary_error_style
fig,axes=plt.subplots(1,2,figsize=(10,5),layout='constrained')
error_cmap=binary_error_style(fig)
for ax,err,title in zip(axes,[err_ce,err_w],['CE test errors','Weighted CE test errors']):
    ax.imshow(err,cmap=error_cmap,vmin=0,vmax=1,interpolation='nearest'); ax.set_title(title); ax.axis('off')
plt.show()
plt.close(fig)
"""),
code('spatial-errors',"""from scipy.ndimage import uniform_filter
ids=split['test_ids']; flat=gt.ravel()
coverage=np.zeros(gt.shape,dtype=float)
for c in range(1,17): coverage+=(gt==c)*uniform_filter((gt==c).astype(float),size=cfg['patch_size'],mode='constant')
values=coverage.ravel()[ids]
for lo,hi in [(0,.3),(.3,.8),(.8,1.00001)]:
    mask=(values>=lo)&(values<hi)
    print('Labeled same-class fraction:',lo,hi,'n=',int(mask.sum()))
    for label,(pred,_) in results.items(): print(label,float(np.mean(pred[mask]==true[mask])) if mask.any() else 'empty')
print('Unlabeled pixels reduce this fraction; it is NOT a complete land-cover purity label.')
"""),md('exercise','## 练习与证据边界\n同一输入换目标类别，重新求Grad-CAM；热区是否变化？这仍不是因果证明。保持相同划分比较CE与加权CE，并报告代价，不只展示上涨的一项。完整训练用准备脚本；演示不加载未知来源旧权重。')]
    return cells


BUILDERS={'00_environment_check.ipynb':environment,'15_imbalance_analysis_teaching.ipynb':imbalance,
          '16_protonet_teaching.ipynb':fewshot,'17_openset_teaching.ipynb':openset}


def build(names=None,execute=True):
    selected=names or list(BUILDERS)
    draft_dir=ROOT/'results/teaching_drafts'; draft_dir.mkdir(parents=True,exist_ok=True)
    for name in selected:
        nb=nbformat.v4.new_notebook(cells=BUILDERS[name]())
        nb.metadata['kernelspec']={'name':'python3','display_name':'HSI-Learning project Python','language':'python'}
        target=ROOT/'notebooks'/name
        if execute:
            # A supplied kernelspec manager binds python3 to the invoking interpreter.
            import sys,json
            from jupyter_client.kernelspec import KernelSpecManager
            with tempfile.TemporaryDirectory() as tmp:
                spec=Path(tmp)/'python3'; spec.mkdir()
                (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                     'display_name':'HSI-Learning','language':'python'}),encoding='utf-8')
                km=KernelManager(kernel_name='python3',kernel_spec_manager=KernelSpecManager(kernel_dirs=[tmp]))
                client=NotebookClient(nb,timeout=1200,km=km,resources={'metadata':{'path':str(ROOT)}})
                client.execute()
            issues = output_issues(nb, name)
            if issues:
                diagnostic = draft_dir/name
                nbformat.write(nb, diagnostic)
                raise RuntimeError(f'Notebook output validation failed: {name}: {issues}. '
                                   f'Diagnostic saved to {diagnostic}; existing notebook preserved.')
            temp=target.with_suffix('.ipynb.tmp'); nbformat.write(nb,temp); os.replace(temp,target)
            print(f'Executed and saved {target}',flush=True)
        else:
            nbformat.write(nb,draft_dir/name)
            print(f'DRAFT ONLY: {draft_dir/name}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft-only',action='store_true')
    parser.add_argument('--only',choices=list(BUILDERS),nargs='+')
    args=parser.parse_args(); build(args.only,not args.draft_only)
