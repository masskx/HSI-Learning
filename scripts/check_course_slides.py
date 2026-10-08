"""Inspect editable PPTX packages; optionally render with installed LibreOffice.

Does not install applications or upload presentations. PDF/PNG rendering is
recorded only after the actual engine succeeds. All-slide PNGs require PyMuPDF.
"""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--only',nargs='+')
    args=parser.parse_args()
    from pptx import Presentation
    decks=sorted((ROOT/'slides/decks').glob('L*.pptx'))
    if args.only: decks=[p for p in decks if p.stem.split('-')[0] in args.only]
    if not decks: raise SystemExit('No generated PPTX decks; run build_course_slides.py first.')
    out=ROOT/'slides/previews'; out.mkdir(parents=True,exist_ok=True)
    executable=shutil.which('soffice') or shutil.which('libreoffice')
    if not executable:
        for p in [Path('C:/Program Files/LibreOffice/program/soffice.exe'),Path('D:/Program Files/LibreOffice/program/soffice.exe')]:
            if p.is_file(): executable=str(p); break
    results=[]
    for deck in decks:
        issues=[]
        with zipfile.ZipFile(deck) as archive:
            corrupt=archive.testzip()
            if corrupt: issues.append('Corrupt ZIP entry: '+corrupt)
        prs=Presentation(deck)
        if not 10<=len(prs.slides)<=18: issues.append('Unexpected slide count')
        for number,slide in enumerate(prs.slides,1):
            texts=[s for s in slide.shapes if s.has_text_frame and s.text.strip()]
            if not texts: issues.append(f'Slide {number}: no editable text')
            if not slide.has_notes_slide or not slide.notes_slide.notes_text_frame.text.strip():
                issues.append(f'Slide {number}: missing speaker notes')
            for s in slide.shapes:
                # Background shapes can lie exactly on the boundary; overflow is not ignored.
                if min(s.left,s.top)<0 or s.left+s.width>prs.slide_width+100 or s.top+s.height>prs.slide_height+100:
                    issues.append(f'Slide {number}: out-of-bounds shape {s.name}')
        row=dict(file=deck.name,slides=len(prs.slides),package_issues=issues,rendered=False,visual_review='pending')
        if args.render:
            if not executable:
                row['render_blocker']='LibreOffice not found. Open in PowerPoint/WPS and export PDF, or install a trusted renderer separately.'
            else:
                try:
                    with tempfile.TemporaryDirectory(prefix='hsi-slide-render-') as tmp:
                        profile=(Path(tmp)/'profile').as_uri()
                        command=[executable,f'-env:UserInstallation={profile}','--headless','--convert-to','pdf','--outdir',tmp,str(deck)]
                        process=subprocess.run(command,capture_output=True,text=True,timeout=180)
                        pdf=Path(tmp)/(deck.stem+'.pdf')
                        if process.returncode!=0 or not pdf.is_file():
                            raise RuntimeError(process.stdout+'\n'+process.stderr)
                        shutil.copyfile(pdf,out/pdf.name)
                    import fitz
                    with fitz.open(out/(deck.stem+'.pdf')) as doc:
                        if len(doc)!=len(prs.slides): raise ValueError('Rendered page count differs')
                        page_dir=out/deck.stem; page_dir.mkdir(exist_ok=True)
                        for i,page in enumerate(doc): page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(page_dir/f'{i+1:02}.png')
                    row['rendered']=True
                except Exception as exc: row['render_blocker']=str(exc)
        results.append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    report=out/'validation.json'
    report.write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Report: {report}. Package checks are NOT visual review.')
    return 1 if any(r['package_issues'] or (args.render and not r['rendered']) for r in results) else 0


if __name__=='__main__': raise SystemExit(main())
