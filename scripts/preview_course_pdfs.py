"""Rasterize user-exported PowerPoint/WPS PDFs for actual visual review.

Export each deck to a PDF with the same stem, then:
    python scripts/preview_course_pdfs.py --pdf-dir slides/previews/pdf --only L01 L09

This is NOT a PPT renderer: the PDF must come from opening/exporting the PPTX
in an actual presentation application. Page-count and age checks cannot prove
that a PDF corresponds to the current PPTX; the exporter must confirm that.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf-dir', type=Path, required=True)
    parser.add_argument('--only', nargs='+', choices=[f'L{i:02}' for i in range(11)])
    args = parser.parse_args()
    import fitz
    from pptx import Presentation

    decks = sorted((ROOT/'slides/decks').glob('L*.pptx'))
    if args.only:
        decks = [p for p in decks if p.stem.split('-')[0] in args.only]
    if not decks:
        raise SystemExit('No matching PPTX decks found.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT/'slides/previews'/f'manual-{stamp}'
    out.mkdir(parents=True)
    records = []
    for deck in decks:
        pdf = args.pdf_dir/(deck.stem+'.pdf')
        row = dict(deck=deck.name, pdf=str(pdf), status='blocked', visual_review='pending',
                   correspondence='requires confirmation by person exporting the PDF')
        try:
            if not pdf.is_file():
                raise FileNotFoundError(f'Export {deck.name} as {pdf.name} first.')
            if pdf.stat().st_mtime < deck.stat().st_mtime:
                raise ValueError('PDF is older than the PPTX; export the current deck again.')
            with fitz.open(pdf) as doc:
                expected = len(Presentation(deck).slides)
                if len(doc) != expected:
                    raise ValueError(f'Expected {expected} pages, found {len(doc)}')
                page_dir = out/deck.stem
                page_dir.mkdir()
                for i, page in enumerate(doc):
                    page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(page_dir/f'{i+1:02}.png')
                row.update(status='rasterized', pages=len(doc), images=str(page_dir),
                           pptx_sha256=digest(deck), pdf_sha256=digest(pdf))
        except Exception as exc:
            row['error'] = str(exc)
        records.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (out/'report.json').write_text(json.dumps(records,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Report: {out / "report.json"}. Rasterized pages still require visual inspection.')
    return 0 if all(r['status']=='rasterized' for r in records) else 1


if __name__ == '__main__':
    raise SystemExit(main())
