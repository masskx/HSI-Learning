"""Read-only readiness checks. A missing artifact is a blocker, not a pass.

Run from the project Python. Writes only a JSON report under results/.
"""
from pathlib import Path
import ast
import json
import re
import sys
import time
from lecture_notebook_validation import output_issues

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    checks=[]
    def record(name,passed,detail=''):
        checks.append(dict(check=name,passed=bool(passed),detail=detail))
        print(('PASS ' if passed else 'FAIL ')+name+': '+detail)
    for p in [* (ROOT/'scripts').glob('*.py'),*(ROOT/'src').rglob('*.py')]:
        try: ast.parse(p.read_text(encoding='utf-8')); record('syntax '+str(p.relative_to(ROOT)),True)
        except SyntaxError as exc: record('syntax '+str(p.relative_to(ROOT)),False,str(exc))
    names=['00_environment_check.ipynb','15_imbalance_analysis_teaching.ipynb','16_protonet_teaching.ipynb','17_openset_teaching.ipynb']
    for name in names:
        p=ROOT/'notebooks'/name
        if not p.exists(): record(name,False,'not built'); continue
        nb=json.loads(p.read_text(encoding='utf-8'))
        code=[c for c in nb['cells'] if c['cell_type']=='code']
        errors=[o for c in code for o in c.get('outputs',[]) if o.get('output_type')=='error']
        source='\n'.join(''.join(c['source']) for c in code)
        glyph=[o for c in code for o in c.get('outputs',[]) if 'missing from font' in ''.join(o.get('text',[]))]
        canonical=any(c.get('id')=='environment' for c in nb['cells'])
        record(name, bool(code) and all(c.get('execution_count') is not None for c in code)
               and not errors and not glyph and canonical,
               'requires canonical cell IDs, all code executed, no errors/font warnings')
        issues = output_issues(nb, name)
        record(name+' embedded figures', not issues, '; '.join(issues) or 'all required plotting cells contain PNG outputs')
        forbidden=['threshold = msp_scores[is_known_test].mean()', 'cam = fmap[0].mean(dim=0)',
                   'evaluate(test_sampler, 20']
        record(name+' protocol guard',not any(s in source for s in forbidden))
    try:
        from hsi_learning.teaching_runs import load_bundle,encode_ids
        from hsi_learning.teaching import prototypes,unknown_scores
        import torch
        import numpy as np
        root=ROOT/'results/teaching_ready'
        for name in ['classifier-ce','classifier-weighted','protonet','openset']:
            t=time.perf_counter()
            try:
                model,cube,gt,names,split,cfg=load_bundle(root/name)
                if name=='openset':
                    classes=cfg['classes']
                    train=encode_ids(model,cube,split['train_ids'],classifier=True)
                    y=torch.tensor([classes.index(int(v)) for v in gt.ravel()[split['train_ids']]])
                    centers=prototypes(train,y,range(len(classes)))
                    te=encode_ids(model,cube,split['test_ids'],classifier=True)
                    with torch.no_grad(): scores=unknown_scores(model.head(te),te,centers)
                    expected=np.load(root/name/'predictions.npz')
                    for key,score in zip(['msp','distance'],scores):
                        np.testing.assert_allclose(score.numpy(),expected[key+'_score'],rtol=1e-5,atol=1e-5)
                record('bundle '+name,True,f'load/verify {time.perf_counter()-t:.2f}s')
            except Exception as exc: record('bundle '+name,False,str(exc))
    except ImportError as exc: record('runtime dependencies',False,str(exc))
    for p in [ROOT/'README.md',*(ROOT/'docs').rglob('*.md')]:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if target.startswith(('https:','http:','#','mailto:')): continue
            record('link '+str(p.relative_to(ROOT))+' '+target,(p.parent/target.split('#')[0]).exists())
    out=ROOT/'results/teaching_readiness.json'; out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(checks,indent=2,ensure_ascii=False),encoding='utf-8')
    return 0 if all(c['passed'] for c in checks) else 1


if __name__=='__main__': raise SystemExit(main())
