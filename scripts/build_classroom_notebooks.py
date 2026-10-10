"""Build canonical classroom demos; run from either root or notebooks directory.

Legacy model notebooks remain historical training examples. New demos do not
silently train: prepare_classroom.py and prepare_teaching_artifacts.py run first.
"""
from pathlib import Path
import argparse
import shutil
from build_lecture_notebooks import BOOT, code, md, build, BUILDERS


def svm():
    return [md('overview','# L02/L03：统一划分的 SVM 基线\n\n协议E：IP、seed42、10/10/80；PCA/scaler仅拟合训练中心。默认载入课前准备的可信本地模型，训练命令为 `python scripts/prepare_classroom.py`。历史80/20版本在 `archive/02_svm_baseline.ipynb`。'),
    code('environment',BOOT),code('protocol',"""import json, joblib, hashlib
ASSETS=ROOT/'results/classroom/svm'
if not (ASSETS/'run_config.json').exists():
    raise FileNotFoundError('Run python scripts/prepare_classroom.py before the lesson')
cfg=json.loads((ASSETS/'run_config.json').read_text())
for filename,sha in json.loads((ASSETS/'hashes.json').read_text()).items():
    assert hashlib.sha256((ASSETS/filename).read_bytes()).hexdigest()==sha, filename
for filename,sha in cfg['data_sha256'].items():
    assert hashlib.sha256((ROOT/'dataset'/filename).read_bytes()).hexdigest()==sha
cube,gt,names=load_hsi_dataset('IP',ROOT/'dataset')
split=dict(np.load(ASSETS/'split.npz'))
from hsi_learning.classroom import classroom_split, classification_summary
for k,v in classroom_split(gt).items(): np.testing.assert_array_equal(v,split[k])
print('Protocol E:',{k:len(v) for k,v in split.items()})
print('Train fit -> validation selection -> frozen test report')
"""),code('metrics-toy',"""from sklearn.metrics import cohen_kappa_score
true=np.repeat([1,2,3],[900,90,10]); predicted=np.ones(1000,dtype=int)
toy=classification_summary(true,predicted,[1,2,3])
print('Confusion rows = truth:',toy['confusion_matrix'])
print('OA, AA, Kappa:',toy['oa'],toy['aa'],toy['kappa'])
assert toy['oa']==.9 and np.isclose(toy['aa'],1/3) and toy['kappa']==0
"""),code('pipeline',"""from hsi_learning.classroom import svm_pipeline
X=cube.reshape(-1,cube.shape[-1]); y=gt.ravel()
raw=joblib.load(ASSETS/'raw.joblib'); pca=joblib.load(ASSETS/'pca12.joblib')
assert raw.named_steps['standardscaler'].n_samples_seen_==len(split['train_ids'])
print(raw)
print('C selected by validation:',raw.named_steps['svc'].C)
print('raw standardized gamma:',raw.named_steps['svc']._gamma)
print('PCA components:',pca.named_steps['pca'].components_.shape)
# Optional short fit; explicit opt-in, not classroom default.
RUN_FIT=False
if RUN_FIT:
    from hsi_learning.classroom import fit_svm_baseline
    raw,_,_,_=fit_svm_baseline(cube,gt,split)
"""),code('kernel-scale',"""print('Float64 exp(-500):',np.exp(-500.))
assert np.exp(-500.)>0
distance2=100.; gamma=.01
np.testing.assert_allclose(np.exp(-gamma*distance2),np.exp(-(gamma/100)*(100*distance2)))
print('Multiply raw features by 10 => squared distances x100')
print('StandardScaler refit can cancel uniform scaling; test the raw kernel separately.')
"""),code('raw-confusion',"""def show_confusion(model,title):
    ids=split['test_ids']; pred=model.predict(X[ids])
    report=classification_summary(y[ids],pred,np.arange(1,17))
    matrix=np.asarray(report['confusion_matrix'])
    normalized=np.divide(matrix,matrix.sum(1,keepdims=True),out=np.zeros_like(matrix,dtype=float),where=matrix.sum(1,keepdims=True)>0)
    fig,ax=plt.subplots(figsize=(9,7),layout='constrained')
    im=ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues')
    ax.set_xticks(range(16),[f'{i}: {n}' for i,n in enumerate(names,1)],rotation=90)
    ax.set_yticks(range(16),[f'{i}: {n}' for i,n in enumerate(names,1)])
    ax.set_xlabel('Predicted class'); ax.set_ylabel('True class'); ax.set_title(title)
    fig.colorbar(im,ax=ax,label='Row-normalized recall'); plt.show()
    # Projection version uses IDs; names/support remain in the table below.
    fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
    im=ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues')
    ax.set_xticks(range(16),range(1,17),fontsize=16)
    ax.set_yticks(range(16),range(1,17),fontsize=16)
    ax.set_xlabel('Predicted class ID',fontsize=16); ax.set_ylabel('True class ID',fontsize=16)
    ax.set_title(title,fontsize=15)
    fig.colorbar(im,ax=ax,label='Row-normalized recall'); plt.show()
    print('Test OA/AA/Kappa:',report['oa'],report['aa'],report['kappa'])
    print('ID / name / support / recall:')
    for i,n,count,r in zip(report['class_ids'],names,report['support'],report['recall']): print(i,n,count,'NA' if r is None else round(r,3))
    return pred,report
raw_pred,raw_report=show_confusion(raw,'SVM: standardized raw bands, protocol E')
"""),code('pca-confusion',"""pca_pred,pca_report=show_confusion(pca,'SVM: train-fitted PCA12 + scaling, protocol E')
print('Same centers; changing features changes the question. No universal PCA superiority claim.')
"""),code('maps',"""from hsi_learning.teaching_plots import maps,binary_error_style
full=np.load(ASSETS/'raw-predictions.npz')['full']
maps(gt,{'GT':gt,'SVM display':np.where(gt>0,full,0)},names,'Display includes training centers; metrics use test only')
errors=np.full(gt.size,np.nan); errors[split['test_ids']]=(raw_pred!=y[split['test_ids']]).astype(float)
fig,ax=plt.subplots(figsize=(6,5),layout='constrained'); cmap=binary_error_style(fig)
ax.imshow(errors.reshape(gt.shape),cmap=cmap,vmin=0,vmax=1); ax.set_title('Test errors only'); ax.axis('off'); plt.show()
"""),code('spatial-split',"""from hsi_learning.classroom import spatial_split,overlap_fraction
spatial=spatial_split(gt,9)
print('Adjacent pair overlap:',overlap_fraction((0,1)))
for key,ids in spatial.items():
    print(key,len(ids),'missing classes:',sorted(set(range(1,17))-set(y[ids])))
print('Spatial protocol changes class coverage, sample budget and task distribution.')
# Advanced assignment: fit an explicit baseline on this new split; do not call
# the score difference a pure leakage effect. Report absent classes first.
"""),md('exit-ticket','## 退出题\n1. 把类别名称排序后，矩阵含义为何改变？\n2. 标准化后的gamma为什么不能沿用原始DN的量级？\n3. 同样的split是否足以证明两个模型输入信息一样？\n\n完成A02，先做填空版本，再迁移到PU。')]

def training():
    return [md('overview','# L05A：一次参数更新发生了什么？\n\n只做一个batch，不启动完整训练。真实演示模型来自 classifier-ce；一次更新使用复制的模型，不改变演示包。'),code('environment',BOOT),
    code('batch',"""from copy import deepcopy
model,cube,gt,names,split,cfg=load_bundle(ROOT/'results/teaching_ready/classifier-ce')
ids=split['train_ids'][:8]; x=patches_at(cube,ids,cfg['patch_size'])
y=torch.tensor([cfg['classes'].index(int(c)) for c in gt.ravel()[ids]])
print('Patch:',x.shape,'original IDs:',gt.ravel()[ids],'model targets:',y)
assert set(y.tolist())<=set(range(16))
"""),code('forward-loss',"""import torch.nn.functional as F
model.eval()
with torch.no_grad(): logits=model(x)
print('Logits:',logits.shape,'predicted indices:',logits.argmax(1))
print('Cross entropy accepts logits:',float(F.cross_entropy(logits,y)))
assert logits.shape==(len(ids),16)
toy=torch.tensor([[2.,1.]]); label=torch.tensor([0])
print('Hand-check -log(softmax([2,1])[0]):',float(F.cross_entropy(toy,label)))
"""),code('one-update',"""learner=deepcopy(model); learner.eval()  # freeze BN/Dropout for this controlled step
optimizer=torch.optim.SGD(learner.parameters(),lr=.001)
before=learner.head[-1].weight.detach().clone()
optimizer.zero_grad(); loss=F.cross_entropy(learner(x),y); loss.backward()
print('Gradient norm:',float(learner.head[-1].weight.grad.norm()))
optimizer.step()
print('Weight change:',float((learner.head[-1].weight.detach()-before).norm()))
assert not torch.equal(before,learner.head[-1].weight)
torch.testing.assert_close(before,model.head[-1].weight)
print('One batch update is not evidence of test improvement.')
"""),code('mode-gradient',"""learner.eval()
assert learner(x).requires_grad  # eval does NOT turn off autograd
with torch.no_grad(): assert not learner(x).requires_grad
learner.train(); print('training flag:',learner.training)
print('train/eval controls BN and Dropout; grad mode controls autograd.')
"""),code('checkpoint',"""import json
history=json.loads((ROOT/'results/teaching_ready/classifier-ce/history.json').read_text())
fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
epochs=range(1,len(history['val_accuracy'])+1)  # best_epoch is 1-based
ax.plot(epochs,history['train_accuracy'],label='train'); ax.plot(epochs,history['val_accuracy'],label='validation')
ax.axvline(cfg['best_epoch'],color='gray',linestyle=':',label=f"selected epoch {cfg['best_epoch']}")
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.legend(); plt.show()
print('Checkpoint selected by validation:',cfg['best_epoch'])
print('Test was not used to choose the checkpoint.')
"""),md('exit-ticket','## 退出题\n哪些量是参数，哪些是超参数？\n漏掉zero_grad会怎样？\n为什么验证精度下降后不能挑测试精度最高的轮次？\n\n下一步：完成A03中的shape表；不要把一次loss下降写成泛化提升。')]

def patches():
    return [md('overview','# L04/L06：Patch坐标与HybridSN张量流\n\n模型使用随机权重，仅验证接口和结构；不将其输出当作训练成绩。'),code('environment',BOOT),
    code('toy-patches',"""toy=np.arange(25,dtype=np.float32).reshape(5,5,1)
ids=np.array([0,4,12,20,24]); patches=patches_at(toy,ids,3)
np.testing.assert_array_equal(patches[:,0,1,1].numpy(),toy.ravel()[ids])
print('Center 12 patch:',patches[2,0])
fig,axes=plt.subplots(1,5,figsize=(12,3),layout='constrained')
for ax,p,i in zip(axes,patches,ids):
    ax.imshow(p[0],cmap='Blues'); ax.set_title(f'Center ID {i}')
    for r in range(3):
        for c in range(3): ax.text(c,r,int(p[0,r,c]),ha='center',va='center')
    ax.axis('off')
plt.show()
try: patches_at(toy,ids,4)
except ValueError: print('Even patch rejected as expected')
else: raise AssertionError('Even patch must be rejected')
"""),code('hybridsn-shapes',"""from hsi_learning.models import HybridSN
net=HybridSN(15,25,16).eval(); batch=torch.zeros(2,15,25,25)
shapes={}; handles=[]
for name in ['conv1','conv2','conv3','conv4','dense1','dense2','dense3']:
    def record(module,args,output,name=name): shapes[name]=tuple(output.shape)
    handles.append(getattr(net,name).register_forward_hook(record))
try:
    with torch.no_grad(): logits=net(batch)
finally:
    for handle in handles: handle.remove()
print(shapes)
assert shapes['conv3']==(2,32,3,19,19)
assert shapes['conv4']==(2,64,17,17)
assert logits.shape==(2,16)
"""),code('reshape-check',"""tensor=torch.arange(2*32*3*19*19).reshape(2,32,3,19,19)
merged=tensor.reshape(2,96,19,19)
assert tensor.numel()==merged.numel()
assert merged[1,2*3+1,5,7]==tensor[1,2,1,5,7]
print('C,D merge:',tensor.shape,'->',merged.shape)
print('First FC params:',sum(p.numel() for p in net.dense1.parameters()))
"""),code('pca-change',"""other=HybridSN(30,25,16).eval()
assert other.conv4[0].in_channels==32*18
with torch.no_grad(): assert other(torch.zeros(2,30,25,25)).shape==(2,16)
print('PCA30 depth: 30->24->20->18; Conv2d input:',other.conv4[0].in_channels)
"""),md('exit-ticket','## 退出题\nreshape是否减少元素数？\n为什么PCA15变30会影响Conv2d通道数？\n如何检查shuffle之后预测回贴位置？\n\n实测模型结果另见协议E的Notebook15；历史HybridSN训练见Notebook03。')]

def convolutions():
    return [md('overview','# L05：沿哪个轴卷积？\n\n用小张量计算局部乘加，再观察真实模型接口。训练过程在10_training_step_classroom中演示。'),code('environment',BOOT),
    code('manual-convolution',"""import torch.nn as nn
x=torch.arange(1,6,dtype=torch.float32).reshape(1,1,5)
conv=nn.Conv1d(1,1,3,bias=False)
with torch.no_grad(): conv.weight.copy_(torch.tensor([[[1.,0.,-1.]]]))
print('Hand result [-2,-2,-2]:',conv(x))
torch.testing.assert_close(conv(x),torch.tensor([[[-2.,-2.,-2.]]]))
"""),code('three-axes',"""one=nn.Conv1d(1,32,7); two=nn.Conv2d(12,32,3); three=nn.Conv3d(1,8,(7,3,3))
print('1D:',one(torch.zeros(2,1,200)).shape)
print('2D:',two(torch.zeros(2,12,9,9)).shape)
print('3D:',three(torch.zeros(2,1,15,9,9)).shape)
assert sum(p.numel() for p in two.parameters())==3488
print('2D kernel sees all input channels:',two.weight.shape)
"""),code('receptive-field',"""r,jump=1,1
for kernel,stride in [(7,1),(2,2),(5,1),(2,2),(3,1)]:
    r+=(kernel-1)*jump; jump*=stride; print('RF / jump:',r,jump)
assert r==26
print('RF includes padding locations; it is not a wavelength range.')
"""),code('current-curves',"""import json
base=ROOT/'results/teaching_ready/classifier-ce'
history=json.loads((base/'history.json').read_text()); cfg=json.loads((base/'run_config.json').read_text())
fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
ax.plot(history['train_accuracy'],label='Train'); ax.plot(history['val_accuracy'],label='Validation')
ax.set_xlabel('Epoch index'); ax.set_ylabel('Accuracy'); ax.legend(); plt.show()
print('Protocol E, train-fitted PCA12, patch9, seed42, budget:',cfg['epochs'])
"""),md('exit-ticket','## 退出题\n改变batch会不会改变参数量？\nConv2d的通道聚合和Conv3d的depth滑动有何区别？\n为什么不能用不同patch、PCA和训练预算的分数证明卷积维数优劣？')]

BUILDERS.update({'02_svm_baseline.ipynb':svm,'10_training_step_classroom.ipynb':training,
                 '11_patch_hybridsn_classroom.ipynb':patches,'12_convolution_classroom.ipynb':convolutions})

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft-only',action='store_true')
    parser.add_argument('--only',nargs='+',choices=list(BUILDERS))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    archive=root/'notebooks/archive/02_svm_baseline.ipynb'
    if not archive.exists():
        archive.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/'notebooks/02_svm_baseline.ipynb',archive)
    build(args.only or ['02_svm_baseline.ipynb','10_training_step_classroom.ipynb',
                       '11_patch_hybridsn_classroom.ipynb','12_convolution_classroom.ipynb'],not args.draft_only)
