"""Build course PPTs locally with the course design system (deck_design.py).

    python scripts/prepare_course_slides.py --install-deps --research

--install-deps explicitly installs requirements-slides.txt into THIS Python.
--research queries public reference pages only (no course uploads). Network
failure is reported and does not prevent building local, course-owned figures.
No git operations, browser uploads, global npm install, or renderer installation.
Render steps: --render uses LibreOffice if present (records render_blocker
otherwise); --export-previews uses desktop PowerPoint COM on Windows to
rasterize every deck page for visual review.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install-deps',action='store_true')
    parser.add_argument('--research',action='store_true')
    parser.add_argument('--export-previews',action='store_true',
                        help='After building, rasterize all decks via PowerPoint COM.')
    args=parser.parse_args()
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    logs=ROOT/'results/slide_builds'/stamp; logs.mkdir(parents=True)
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    report={'python':sys.executable,'steps':[],'visual_review':'pending'}
    def run(label,argv,required=True,executable=None):
        command = [executable or sys.executable, *argv]
        print(f'{label}: running; log {logs/(label+".log")}',flush=True)
        with (logs/(label+'.log')).open('w',encoding='utf-8') as output:
            try:
                result=subprocess.run(command,cwd=ROOT,env=env,
                                      stdout=output,stderr=subprocess.STDOUT)
                returncode = result.returncode
            except OSError as exc:
                output.write(f'Unable to start command: {exc}\n')
                returncode = 1
        report['steps'].append({'label':label,'returncode':returncode,'command':command})
        (logs/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(f'{label}: '+('passed' if returncode==0 else f'blocked/failed ({returncode})'),flush=True)
        if required and returncode: raise SystemExit(returncode)
    if args.install_deps:
        uv = shutil.which('uv')
        if uv:
            run('dependencies',['pip','install','--python',sys.executable,
                                '-r',str(ROOT/'requirements-slides.txt')],executable=uv)
        elif importlib.util.find_spec('pip') is not None:
            run('dependencies',['-m','pip','install','-r','requirements-slides.txt'])
        else:
            raise SystemExit('Neither uv on PATH nor pip in this Python is available. '
                             'Install uv or bootstrap pip in the project environment, then retry.')
    if args.research:
        run('source-discovery',['scripts/research_slide_images.py'],required=False)
    run('pilot-decks',['scripts/build_course_slides.py','--only','L01','L09'])
    run('pilot-package-check',['scripts/check_course_slides.py','--only','L01','L09'])
    run('pilot-render',['scripts/check_course_slides.py','--only','L01','L09','--render'],required=False)
    run('all-decks',['scripts/build_course_slides.py'])
    run('all-package-check',['scripts/check_course_slides.py'])
    run('all-render',['scripts/check_course_slides.py','--render'],required=False)
    if args.export_previews:
        run('com-preview-export',['powershell','-NoProfile','-File',
            'scripts/export_deck_previews.ps1'],required=False)
    print(f'Build report: {logs/"report.json"}')
    print('PPTX directory: slides/decks. Visual review and external-image permissions remain separate gates.')
    return 0


if __name__=='__main__': raise SystemExit(main())
