"""Build course decks using the bundled Artifact Tool JS engine."""
from pathlib import Path
import argparse,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--only',nargs='+'); args=p.parse_args()
    runtime=Path(os.environ.get('HSI_RUNTIME_ROOT',Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies'))
    node=runtime/'node/bin/node.exe'
    if not node.exists(): raise SystemExit('Set HSI_RUNTIME_ROOT to the bundled dependencies directory.')
    build=ROOT/'results/slide_revision_build'; build.mkdir(parents=True,exist_ok=True)
    link=build/'node_modules'; packages=runtime/'node/node_modules'
    if not link.exists():
        subprocess.run(['powershell','-NoProfile','-Command',f"New-Item -ItemType Junction -Path '{link}' -Target '{packages}' | Out-Null"],check=True)
    content=build/'content.json'
    subprocess.run([sys.executable,str(ROOT/'scripts/export_slide_content.py'),'--output',str(content)],check=True,cwd=ROOT)
    subprocess.run([str(node),str(ROOT/'scripts/build_course_slides.mjs'),str(content),*(args.only or [])],check=True,cwd=ROOT)
    return 0
if __name__=='__main__': raise SystemExit(main())
