"""Run lecture preparation with fail-fast logs; never install, commit or push.

    python scripts/prepare_recording.py --stage test
    python scripts/prepare_recording.py --stage smoke
    python scripts/prepare_recording.py --stage all

Full preparation can take time. Each child is serial, with a preserved log.
A passed automated check does not replace visual inspection or a trial recording.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    'test': ['-m', 'unittest', 'discover', '-s', 'tests', '-v'],
    'smoke': ['scripts/prepare_teaching_artifacts.py', '--quick', '--output-dir', 'results/teaching_smoke'],
    'assets': ['scripts/prepare_teaching_artifacts.py'],
    'notebooks': ['scripts/build_lecture_notebooks.py'],
    'verify': ['scripts/check_teaching_ready.py'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=[*STAGES, 'all'], default='test')
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = ROOT / 'results' / 'recording_checks' / stamp
    directory.mkdir(parents=True)
    env = dict(os.environ, PYTHONIOENCODING='utf-8', MPLBACKEND='Agg')
    env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    report = {'python': sys.executable, 'started_utc': stamp, 'stages': [],
              'visual_review': 'pending', 'trial_recording': 'pending'}
    selected = list(STAGES) if args.stage == 'all' else [args.stage]
    result = 0
    for stage in selected:
        command = [sys.executable, *STAGES[stage]]
        log_path = directory / f'{stage}.log'
        print(f'Running {stage}; log: {log_path}', flush=True)
        try:
            with log_path.open('w', encoding='utf-8') as log:
                completed = subprocess.run(command, cwd=ROOT, env=env,
                                           stdout=log, stderr=subprocess.STDOUT)
            result = completed.returncode
            detail = ''
        except OSError as exc:
            result, detail = 1, str(exc)
        report['stages'].append({'stage': stage, 'command': command,
                                 'returncode': result, 'log': log_path.name, 'detail': detail})
        (directory / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if result:
            print(f'FAILED {stage}; subsequent stages were NOT executed. Read {log_path}', flush=True)
            break
        print(f'PASSED {stage}', flush=True)
    print(f'Report: {directory / "report.json"}')
    return result


if __name__ == '__main__':
    raise SystemExit(main())
