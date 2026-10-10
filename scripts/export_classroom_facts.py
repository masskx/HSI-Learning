"""Export current local run facts for slides; never invent absent results."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hsi_learning.teaching_runs import load_bundle,encode_ids
from hsi_learning.classroom import classification_summary
from hsi_learning.utils import save_json

def main():
    torch.set_num_threads(2)
    facts={'status':'verified-local-execution','protocol':'E','seed':42,'rates':[.1,.1,.8],
           'preprocessing_fit':'training centers only','results':{},'input_hashes':{}}
    for label in ['classifier-ce','classifier-weighted','openset','protonet']:
        base=ROOT/'results/teaching_ready'/label
        model,cube,gt,names,split,cfg=load_bundle(base)
        metrics=json.loads((base/'metrics.json').read_text())
        if label.startswith('classifier'):
            emb=encode_ids(model,cube,split['test_ids'],classifier=True)
            with torch.no_grad(): pred=np.asarray(cfg['classes'])[model.head(emb).argmax(1).numpy()]
            metrics.update(classification_summary(gt.ravel()[split['test_ids']],pred,cfg['classes']))
        facts['results'][label]={'config':cfg,'metrics':metrics,'counts':{k:len(v) for k,v in split.items() if k.endswith('_ids')}}
        facts['input_hashes'][label]=hashlib.sha256((base/'hashes.json').read_bytes()).hexdigest()
    base=ROOT/'results/classroom/svm'
    facts['results']['svm']=json.loads((base/'metrics.json').read_text())
    facts['input_hashes']['svm']=hashlib.sha256((base/'metrics.json').read_bytes()).hexdigest()
    save_json(facts,ROOT/'artifacts/teaching/classroom-facts.json')
    print('Exported verified facts with per-run configuration and input hashes')
if __name__=='__main__': main()
