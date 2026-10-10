"""Execute classroom notebooks from notebooks/; check historical path bootstraps.

Does not rerun long historical training. Saves reports and diagnostic copies.
"""
from pathlib import Path
import json, sys, tempfile, os
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from lecture_notebook_validation import EXPECTED_PNGS, output_issues
ROOT=Path(__file__).resolve().parents[1]

def execute(nb,cwd):
    with tempfile.TemporaryDirectory() as tmp:
        spec=Path(tmp)/'python3'; spec.mkdir()
        (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
            'display_name':'HSI-Learning','language':'python'}))
        km=KernelManager(kernel_name='python3',kernel_spec_manager=KernelSpecManager(kernel_dirs=[tmp]))
        NotebookClient(nb,timeout=1200,km=km,resources={'metadata':{'path':str(cwd)}}).execute()
    return nb

def main():
    out=ROOT/'results/classroom_verification';out.mkdir(parents=True,exist_ok=True);records=[]
    name='01_data_reading_and_visualization.ipynb'; target=ROOT/'notebooks'/name
    nb=execute(nbformat.read(target,as_version=4),ROOT)
    temp=target.with_suffix('.ipynb.tmp');nbformat.write(nb,temp);os.replace(temp,target)
    print('PASS root:',name,flush=True);records.append(dict(name=name,cwd='root',passed=True))
    for name in [name,*EXPECTED_PNGS]:
        nb=execute(nbformat.read(ROOT/'notebooks'/name,as_version=4),ROOT/'notebooks')
        issues=output_issues(nb,name) if name in EXPECTED_PNGS else []
        nbformat.write(nb,out/name)
        if issues: raise RuntimeError(f'{name}: {issues}')
        print('PASS notebooks/:',name,flush=True);records.append(dict(name=name,cwd='notebooks',passed=True))
    for target in sorted((ROOT/'notebooks').glob('0[3-9]_*.ipynb')):
        legacy=nbformat.read(target,as_version=4)
        boot=next(c for c in legacy.cells if c.get('id')=='project-root')
        execute(nbformat.v4.new_notebook(cells=[boot]),ROOT/'notebooks')
        records.append(dict(name=target.name,cwd='notebooks',scope='path bootstrap only; historical training not rerun',passed=True))
    (out/'execution-report.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    print('Saved execution-report.json; historical training is outside this check.',flush=True)
if __name__=='__main__':main()
