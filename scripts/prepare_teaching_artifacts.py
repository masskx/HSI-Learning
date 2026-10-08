"""Prepare lecture demos serially; keep logs and stop on failure (no silent fallback)."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/teaching_ready')
    parser.add_argument('--quick', action='store_true', help='Smoke runs, not reference results.')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), PYTHONIOENCODING='utf-8', MPLBACKEND='Agg')
    manifest = {}
    jobs = [('classifier-ce', 'classifier', ['--loss','ce']),
            ('classifier-weighted', 'classifier', ['--loss','weighted']),
            ('protonet', 'protonet', []), ('openset','openset',[])]
    for name, task, extra in jobs:
        out = args.output_dir/name
        command = [sys.executable,'-m','hsi_learning.teaching_runs','--task',task,
                   '--output-dir',str(out),'--epochs','2' if args.quick else '20',
                   '--episodes','5' if args.quick else '200',
                   '--eval-episodes','3' if args.quick else '30',*extra]
        if not (out/'hashes.json').exists():
            with (args.output_dir/f'{name}.log').open('w',encoding='utf-8') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        # Check before declaring reusable: never resume by blindly trusting file existence.
        sys.path.insert(0,str(ROOT/'src'))
        from hsi_learning.teaching_runs import load_bundle
        _, _, _, _, _, config = load_bundle(out)
        # Do not promote an old smoke run (or another loss) to a reference run.
        expected = {'task': task, 'seed': 42, 'components': 12, 'patch_size': 9}
        if task == 'protonet':
            expected.update(mode='class-disjoint', episodes=5 if args.quick else 200,
                            eval_episodes=3 if args.quick else 30)
        else:
            expected.update(epochs=2 if args.quick else 20,
                            loss='weighted' if name == 'classifier-weighted' else 'ce')
        mismatch = {key: (config.get(key), value) for key, value in expected.items()
                    if config.get(key) != value}
        if mismatch:
            raise ValueError(f'Existing bundle has different settings: {out}: {mismatch}. '
                             'Choose another --output-dir; nothing was overwritten.')
        manifest[name] = {'path':str(out.relative_to(args.output_dir)),
                          'command':subprocess.list2cmdline(command),'smoke_only':args.quick}
        print(f'{name}: verified',flush=True)
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
