"""Upgrade legacy notebook root detection without training or clearing outputs.

Use --apply deliberately; initially emits an inventory. Does not fix notebook
algorithms, and preserves the original cells/outputs except for a root bootstrap
and explicit path substitutions. Outputs affected by algorithm changes must be
regenerated separately by build_lecture_notebooks.py.
"""
import argparse
import ast
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parents[1]
BOOT='''from pathlib import Path
import os, sys
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p/'src/hsi_learning').is_dir() and (p/'dataset').is_dir()), None)
if ROOT is None: raise RuntimeError('Start inside the HSI-Learning repository')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
print('Notebook project root:', ROOT)
'''


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--apply',action='store_true'); args=p.parse_args()
    for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
        if path.name.startswith(('00_','15_','16_','17_')): continue
        nb=nbformat.read(path,as_version=4)
        if any(c.get('id')=='project-root' for c in nb.cells): continue
        first=next((i for i,c in enumerate(nb.cells) if c.cell_type=='code'),len(nb.cells))
        cell=nbformat.v4.new_code_cell(BOOT); cell.id='project-root'
        nb.cells.insert(first,cell)
        print(('APPLY ' if args.apply else 'WOULD ADD ROOT BOOTSTRAP ')+path.name)
        if args.apply: nbformat.write(nb,path)


if __name__=='__main__': main()
