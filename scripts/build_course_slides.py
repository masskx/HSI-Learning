"""Build the classroom decks.

Default engine: pptxgenjs (scripts/build_course_slides.js) -- needs only Node.js and
`npm install` in slides/. The previous design is still available with
`--engine artifact-tool` (scripts/build_course_slides.mjs, Codex Artifact Tool runtime).
"""
from pathlib import Path
import argparse,os,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]

def artifact_tool(content,only):
    runtime=Path(os.environ.get('HSI_RUNTIME_ROOT',Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies'))
    node=runtime/'node/bin/node.exe'
    if not node.exists(): raise SystemExit('Set HSI_RUNTIME_ROOT to the bundled dependencies directory.')
    link=content.parent/'node_modules'; packages=runtime/'node/node_modules'
    if not link.exists():
        subprocess.run(['powershell','-NoProfile','-Command',f"New-Item -ItemType Junction -Path '{link}' -Target '{packages}' | Out-Null"],check=True)
    subprocess.run([str(node),str(ROOT/'scripts/build_course_slides.mjs'),str(content),*only],check=True,cwd=ROOT)

def pptxgenjs(content,only):
    node=shutil.which('node')
    if not node: raise SystemExit('Node.js not found; install Node 18+ and run `npm install` in slides/.')
    if not (ROOT/'slides/node_modules/pptxgenjs').exists():
        npm=shutil.which('npm') or shutil.which('npm.cmd')
        if not npm: raise SystemExit('Run `npm install` in slides/ first.')
        subprocess.run([npm,'install','--no-audit','--no-fund'],check=True,cwd=ROOT/'slides')
    subprocess.run([node,str(ROOT/'scripts/build_course_slides.js'),str(content),*only],check=True,cwd=ROOT)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--only',nargs='+')
    p.add_argument('--engine',choices=['pptxgenjs','artifact-tool'],default='pptxgenjs'); args=p.parse_args()
    build=ROOT/'results/slide_revision_build'; build.mkdir(parents=True,exist_ok=True)
    content=build/'content.json'
    subprocess.run([sys.executable,str(ROOT/'scripts/export_slide_content.py'),'--output',str(content)],check=True,cwd=ROOT)
    (pptxgenjs if args.engine=='pptxgenjs' else artifact_tool)(content,args.only or [])
    return 0
if __name__=='__main__': raise SystemExit(main())
