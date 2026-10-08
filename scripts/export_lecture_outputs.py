"""Export existing lecture notebook images for visual review; no kernel/training.

    python scripts/export_lecture_outputs.py

Writes an HTML contact sheet, PNGs and a JSON inventory into a NEW review
folder. This exports actual embedded outputs, never regenerates charts.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ['00_environment_check.ipynb', '15_imbalance_analysis_teaching.ipynb',
             '16_protonet_teaching.ipynb', '17_openset_teaching.ipynb']


def text(value):
    return ''.join(value) if isinstance(value, list) else str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = args.output_dir or ROOT/'results/lecture_visual_review'/stamp
    if out.exists() and any(out.iterdir()):
        parser.error('Review directory must be empty; existing outputs are not overwritten.')
    out.mkdir(parents=True, exist_ok=True)
    entries, problems, blocks = [], [], []
    for name in NOTEBOOKS:
        path = ROOT/'notebooks'/name
        if not path.exists():
            problems.append(f'{name}: missing notebook'); continue
        nb = json.loads(path.read_text(encoding='utf-8'))
        heading = ''
        image_count = 0
        for index, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'markdown':
                heading = text(cell.get('source', '')).split('\n')[0]
                continue
            if cell.get('execution_count') is None:
                problems.append(f'{name} cell {index}: not executed')
            cell_id = re.sub(r'[^a-zA-Z0-9_-]', '_', cell.get('id', f'cell-{index}'))
            for output_index, output in enumerate(cell.get('outputs', [])):
                if output.get('output_type') == 'error':
                    problems.append(f'{name} {cell_id}: {output.get("ename")}: {output.get("evalue")}')
                png = output.get('data', {}).get('image/png')
                if not png:
                    continue
                blob = base64.b64decode(text(png), validate=False)
                if not blob.startswith(b'\x89PNG\r\n\x1a\n'):
                    raise ValueError(f'Invalid PNG output in {name}/{cell_id}')
                filename = f'{Path(name).stem}--{cell_id}--{output_index}.png'
                (out/filename).write_bytes(blob)
                width = int.from_bytes(blob[16:20], 'big')
                height = int.from_bytes(blob[20:24], 'big')
                record = dict(notebook=name, cell_id=cell_id, heading=heading, file=filename,
                              width=width, height=height, sha256=hashlib.sha256(blob).hexdigest(),
                              review='pending')
                entries.append(record); image_count += 1
                caption = html.escape(f'{name} / {cell_id} / {heading} ({width} x {height})')
                blocks.append(f'<section><h2>{caption}</h2><a href="{filename}"><img src="{filename}" alt="{caption}"></a>'
                              '<p>Review: labels, clipping, legend, class mapping, errors vs claims.</p></section>')
        print(f'{name}: exported {image_count} images', flush=True)
        if image_count == 0:
            problems.append(f'{name}: no embedded PNG outputs')
    inventory = dict(exported_utc=stamp, figures=entries, problems=problems,
                     status='exported-not-visually-verified')
    (out/'inventory.json').write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding='utf-8')
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Lecture output review</title>'
            '<style>body{font-family:system-ui;margin:2rem;background:#eee;color:#111}'
            'section{background:white;margin:2rem 0;padding:1rem}img{max-width:100%;height:auto}'
            'h2{font-size:1rem;overflow-wrap:anywhere}</style>'
            '<h1>Embedded notebook figures: review pending</h1>' + ''.join(blocks) + '</html>')
    (out/'index.html').write_text(page, encoding='utf-8')
    print(f'Review index: {out/"index.html"}')
    print(f'Inventory: {out/"inventory.json"}')
    for issue in problems:
        print('ISSUE:', issue)
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
