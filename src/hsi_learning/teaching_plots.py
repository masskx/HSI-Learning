"""Lecture figures: explicit labels, no interpretation inferred from a pretty map."""
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix

from .teaching import patches_at, grad_cam, prototypes, unknown_scores, rejection_predictions
from .teaching_runs import encode_ids


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.unicode_minus':False,
                         'figure.figsize':(10,4),'figure.dpi':110})


def class_palette():
    colors = plt.get_cmap('nipy_spectral')(np.arange(17)/16)
    colors[0] = [0,0,0,1]
    colors = np.vstack([colors,[1,1,1,1]])
    return ListedColormap(colors), BoundaryNorm(np.arange(-.5,18.5),18)


def maps(gt, panels, names, title):
    """0=background; 17=unknown; all other colors retain their GT identity."""
    cmap,norm=class_palette()
    fig,axes=plt.subplots(1,len(panels),figsize=(5*len(panels),5),layout='constrained')
    for ax,(label,array) in zip(np.atleast_1d(axes),panels.items()):
        im=ax.imshow(array,cmap=cmap,norm=norm,interpolation='nearest')
        ax.set_title(label); ax.set_axis_off()
    fig.suptitle(title)
    handles = [Patch(facecolor='black', edgecolor='gray', label='0: unlabeled background')]
    if any(np.any(np.asarray(array) == 17) for array in panels.values()):
        handles.append(Patch(facecolor='white', edgecolor='gray', label='17: unknown / rejected'))
    fig.legend(handles=handles, loc='outside lower center', ncol=len(handles))
    plt.show()
    plt.close(fig)
    print('Class IDs:',dict(enumerate(names,1)))


@torch.no_grad()
def classifier_maps(model,cube,gt,split,cfg):
    ids=np.arange(gt.size)
    emb=encode_ids(model,cube,ids,cfg['patch_size'],classifier=True)
    logits=model.head(emb)
    pred=np.asarray(cfg['classes'])[logits.argmax(1).numpy()].reshape(gt.shape)
    shown=np.where(gt>0,pred,0)
    err=np.full(gt.size,np.nan); test=split['test_ids']
    err[test]=(pred.ravel()[test]!=gt.ravel()[test]).astype(float)
    return shown,err.reshape(gt.shape)


@torch.no_grad()
def openset_maps(model,cube,gt,split,cfg):
    train_ids=split['train_ids']; classes=cfg['classes']
    emb=encode_ids(model,cube,train_ids,cfg['patch_size'],classifier=True)
    labels=torch.tensor([classes.index(int(c)) for c in gt.ravel()[train_ids]])
    centers=prototypes(emb,labels,range(len(classes)))
    ids=np.arange(gt.size); all_emb=encode_ids(model,cube,ids,cfg['patch_size'],classifier=True)
    logits=model.head(all_emb); pred=np.asarray(classes)[logits.argmax(1).numpy()]
    scores=unknown_scores(logits,all_emb,centers)
    output={}
    for method,score in zip(['msp','distance'],scores):
        output[method]=rejection_predictions(pred,score.numpy(),cfg['thresholds'][method],17).reshape(gt.shape)
    target=gt.copy(); target[np.isin(gt,cfg['unknown_classes'])]=17
    # GT used ONLY after predictions for display masking and error analysis.
    panels={'Open-set GT':target,**{k:np.where(gt>0,v,0) for k,v in output.items()}}
    return panels,output


def binary_error_style(fig):
    """Common legend: only explicitly evaluated pixels are assigned 0/1."""
    cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad('#dddddd')
    fig.legend(handles=[
        Patch(facecolor=cmap(1.0), label='1: specified error'),
        Patch(facecolor=cmap(0.0), edgecolor='gray', label='0: other evaluated pixel'),
        Patch(facecolor='#dddddd', label='Not evaluated'),
    ], loc='outside lower center', ncol=3)
    return cmap


def error_panels(gt,outputs,split,cfg):
    """Facet binary errors rather than encode five categories with competing hues."""
    test=split['test_ids']; true=gt.ravel()[test]; known=np.isin(true,cfg['classes'])
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    error_cmap = binary_error_style(fig)
    for row,(method,pred_map) in enumerate(outputs.items()):
        pred=pred_map.ravel()[test]
        masks=[known&(pred==17),~known&(pred!=17),known&(pred!=17)&(pred!=true)]
        for ax,label,mask in zip(axes[row],['Known false rejection','Unknown miss','Known misclassification'],masks):
            canvas=np.full(gt.size,np.nan); canvas[test]=mask.astype(float)
            ax.imshow(canvas.reshape(gt.shape),cmap=error_cmap,vmin=0,vmax=1,interpolation='nearest')
            ax.set_title(f'{method}: {label}\nn={int(mask.sum())}'); ax.set_axis_off()
    plt.show()
    plt.close(fig)


def episode_figure(gt,episode,names):
    fig,axes=plt.subplots(1,len(episode.classes),figsize=(3*len(episode.classes),3),layout='constrained')
    for local,(ax,c) in enumerate(zip(np.atleast_1d(axes),episode.classes)):
        ax.imshow(gt==c,cmap='gray',interpolation='nearest')
        for ids,marker,label in [(episode.support_ids[episode.support_y==local],'o','Support'),
                                  (episode.query_ids[episode.query_y==local],'x','Query')]:
            r,col=np.unravel_index(ids,gt.shape)
            ax.scatter(col,r,s=36,marker=marker,label=label,facecolors='none' if marker=='o' else None,
                       edgecolors='#2a78d6' if marker=='o' else None,color=None if marker=='o' else '#eb6834')
        ax.set_title(f'{c}: {names[c-1]}',fontsize=9); ax.set_axis_off()
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2)
    plt.show()
    plt.close(fig)
    # Do not return Figure: a notebook's final expression would display it twice.


def cam_figure(model,cube,gt,test_ids,patch_size=9,classes=None,names=None):
    """Deterministic examples, not cherry-picked heatmaps: correct/error/rare.

    Selection uses predictions and labels only for post-test illustration.
    Targets are the model's predicted logits; report original class IDs explicitly.
    """
    classes = np.asarray(classes if classes is not None else np.arange(1,17))
    ids = np.asarray(test_ids)
    embeddings = encode_ids(model,cube,ids,patch_size,classifier=True)
    with torch.no_grad():
        prediction = classes[model.head(embeddings).argmax(1).numpy()]
    truth = gt.ravel()[ids]
    counts = np.bincount(gt.ravel())
    rare_class = min(np.unique(truth), key=lambda c: counts[c])
    groups = [('Correct', np.flatnonzero(prediction == truth)),
              ('Misclassified', np.flatnonzero(prediction != truth)),
              ('Rare-class example', np.flatnonzero(truth == rare_class))]
    selected, captions = [], []
    for label, candidates in groups:
        candidate = next((int(i) for i in candidates if int(i) not in selected), None)
        if candidate is None:
            print(f'No distinct sample for {label}; not inventing one.')
            continue
        selected.append(candidate); captions.append(label)
    if not selected:
        raise ValueError('No test samples for Grad-CAM')
    chosen = ids[selected]
    x=patches_at(cube,chosen,patch_size)
    cams,logits=grad_cam(model,model.encoder.features[7],x)
    fig,axes=plt.subplots(2,len(x),figsize=(4*len(x),7),layout='constrained',squeeze=False)
    for i,(idx,label) in enumerate(zip(selected,captions)):
        original_prediction = int(classes[int(logits[i].argmax())])
        true_id = int(truth[idx])
        r,c = np.unravel_index(chosen[i],gt.shape)
        axes[0,i].imshow(x[i,0],cmap='gray',interpolation='nearest')
        axes[0,i].set_title(f'{label}; pixel ({r},{c})\nGT={true_id}; predicted/target={original_prediction}',fontsize=10)
        im=axes[1,i].imshow(cams[i],cmap='Blues',vmin=0,vmax=1,interpolation='nearest')
        axes[1,i].set_title('Gradient-weighted CAM')
        axes[0,i].set_axis_off(); axes[1,i].set_axis_off()
        print({'case':label,'pixel':(int(r),int(c)),'true_id':true_id,'target_id':original_prediction,
               'true_name':names[true_id-1] if names else str(true_id)})
    fig.suptitle('Top: PC 1 input; bottom: CAM for predicted class (original label IDs)')
    fig.colorbar(im,ax=axes[1].tolist(),label='Normalized response')
    plt.show()
    plt.close(fig)
    print('Zero CAM is possible. These are class-conditioned gradients, not a causal proof.')
    return cams,logits
