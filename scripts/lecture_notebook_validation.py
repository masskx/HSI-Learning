"""Shared output checks for lecture notebook building and final acceptance.

These checks inspect actual saved outputs, not source text containing plt.show().
They do not replace visual inspection of labels, layout and scientific meaning.
"""
from __future__ import annotations

import base64

EXPECTED_PNGS = {
    '02_svm_baseline.ipynb': {'raw-confusion': 2, 'pca-confusion': 2, 'maps': 2},
    '10_training_step_classroom.ipynb': {'checkpoint': 1},
    '11_patch_hybridsn_classroom.ipynb': {'toy-patches': 1},
    '12_convolution_classroom.ipynb': {'current-curves': 1},
    '00_environment_check.ipynb': {'dataset': 1},
    '15_imbalance_analysis_teaching.ipynb': {
        'loss-comparison': 1, 'gradcam': 1, 'target-comparison': 1, 'maps': 2,
    },
    '16_protonet_teaching.ipynb': {'episode': 1, 'evaluation': 1, 'maps': 1},
    '17_openset_teaching.ipynb': {'evaluation': 1, 'maps': 2, 'threshold-exercise': 2},
}


def joined(value):
    return ''.join(value) if isinstance(value, list) else str(value)


def output_issues(notebook, name):
    """Return actionable failures; missing/invalid PNGs fail the build."""
    issues = []
    cells = notebook.get('cells', [])
    ids = [c.get('id') for c in cells]
    if len(ids) != len(set(ids)):
        issues.append('duplicate cell IDs')
    expected = EXPECTED_PNGS[name]
    found = {}
    for cell in cells:
        if cell.get('cell_type') != 'code':
            continue
        cell_id = cell.get('id', '<missing>')
        if cell.get('execution_count') is None:
            issues.append(f'{cell_id}: not executed')
        png_count = 0
        for output in cell.get('outputs', []):
            if output.get('output_type') == 'error':
                issues.append(f'{cell_id}: execution error: {output.get("ename")}')
            message = joined(output.get('text', '')).lower()
            if 'missing from font' in message or 'non-interactive' in message or 'cannot be shown' in message:
                issues.append(f'{cell_id}: font or non-interactive-backend warning')
            png = output.get('data', {}).get('image/png')
            if png:
                try:
                    blob = base64.b64decode(''.join(joined(png).split()), validate=True)
                    valid = (len(blob) >= 24 and blob[:8] == b'\x89PNG\r\n\x1a\n'
                             and blob[12:16] == b'IHDR'
                             and int.from_bytes(blob[16:20], 'big') > 0
                             and int.from_bytes(blob[20:24], 'big') > 0)
                except (ValueError, TypeError):
                    valid = False
                if valid:
                    png_count += 1
                else:
                    issues.append(f'{cell_id}: malformed embedded PNG')
        found[cell_id] = png_count
    for cell_id, minimum in expected.items():
        count = found.get(cell_id, 0)
        if count < minimum:
            issues.append(f'{cell_id}: expected at least {minimum} embedded PNG(s), got {count}')
    return issues
