"""Resolve course-owned notebook figures and existing formula assets for JS authoring."""
from pathlib import Path
import argparse,base64,hashlib,json,sys,zipfile,shutil,posixpath
import xml.etree.ElementTree as E
import io,re
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'slides/src'))

def redraw_historical_training(cell):
    """Re-layout saved logs without training again or changing their values."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    logs=''.join(''.join(o.get('text',[])) for o in cell.get('outputs',[]) if o.get('output_type')=='stream')
    train=re.findall(r'epoch\s+(\d+)/20\s+loss:([\d.eE+-]+)\s+acc:([\d.eE+-]+)',logs)
    val=re.findall(r'epoch\s+(\d+)/20\s+val_acc:([\d.eE+-]+)',logs)
    if len(train)!=20 or len(val)!=4:
        raise ValueError('Historical training logs must contain 20 train and 4 validation observations')
    with plt.rc_context({'font.size':14,'text.color':'#152E50','axes.labelcolor':'#152E50','xtick.color':'#5C6B7A','ytick.color':'#5C6B7A'}):
        fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        epochs=[int(t[0]) for t in train]
        axs[0].plot(epochs,[float(t[1]) for t in train],color='#315DAB',linewidth=2.5)
        axs[0].set_title('Historical logged loss',loc='left',fontweight='bold')
        axs[0].set_ylabel('Logged loss')
        axs[1].plot(epochs,[float(t[2]) for t in train],color='#315DAB',linewidth=2.5,label='Train')
        axs[1].plot([int(t[0]) for t in val],[float(t[1]) for t in val],color='#438C91',marker='o',linewidth=2,label='Validation')
        axs[1].set_title('Historical accuracy',loc='left',fontweight='bold')
        axs[1].set_ylabel('Accuracy'); axs[1].set_ylim(0,1.05); axs[1].legend(frameon=False)
        for ax in axs:
            ax.set_xlabel('Epoch'); ax.set_xticks([0,5,10,15,19])
            ax.spines[['top','right']].set_visible(False)
            ax.grid(axis='y',color='#D9E2E9',alpha=.6)
        buffer=io.BytesIO(); fig.savefig(buffer,format='png',dpi=160); plt.close(fig)
    return buffer.getvalue()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    reference=ROOT/'slides/assets/reference-formulas'
    reference.mkdir(parents=True,exist_ok=True)
    reference_manifest=reference/'manifest.json'
    if reference_manifest.exists():
        formula_assets={k:str(reference/v) for k,v in json.loads(reference_manifest.read_text(encoding='utf-8')).items()}
    else:
        backup=ROOT/'results/slide_revision_original'; backup.mkdir(parents=True,exist_ok=True)
        for deck in (ROOT/'slides/decks').glob('*.pptx'):
            if not (backup/deck.name).exists(): shutil.copy2(deck,backup/deck.name)
        manifest=backup/'lesson-manifest.json'
        if not manifest.exists(): shutil.copy2(ROOT/'slides/lesson-manifest.json',manifest)
        formula_assets={}
        ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
        for old in json.loads(manifest.read_text(encoding="utf-8"))['decks']:
            deck=backup/Path(old['path'].replace('\\','/')).name
            with zipfile.ZipFile(deck) as z:
                for index,entry in enumerate(old['pages'],1):
                    if not entry.get('math') or entry.get('image'): continue
                    slide=E.fromstring(z.read(f'ppt/slides/slide{index}.xml'))
                    rels=E.fromstring(z.read(f'ppt/slides/_rels/slide{index}.xml.rels'))
                    mapping={r.get('Id'):r.get('Target') for r in rels}
                    blips=slide.findall('.//a:blip',ns)
                    if not blips: continue
                    target=mapping[blips[0].get('{'+ns['r']+'}embed')]
                    blob=z.read(posixpath.normpath(posixpath.join('ppt/slides',target)))
                    asset=backup/(hashlib.sha256(blob).hexdigest()[:16]+'.png'); asset.write_bytes(blob)
                    formula_assets[entry['math']]=str(asset)
        saved={}
        for formula,asset in formula_assets.items():
            destination=reference/Path(asset).name
            shutil.copy2(asset,destination)
            saved[formula]=destination.name
            formula_assets[formula]=str(destination)
        reference_manifest.write_text(json.dumps(saved,ensure_ascii=False,indent=2),encoding='utf-8')
    from course_content import LESSONS,SOURCES
    assets=ROOT/'slides/assets/generated'; assets.mkdir(parents=True,exist_ok=True)
    records=[]
    for lesson in LESSONS:
        for entry in lesson['pages']:
            if entry.get('image'):
                name,cell_id,ordinal=entry['image']; nb=json.loads((ROOT/'notebooks'/name).read_text(encoding='utf-8'))
                cells=[c for c in nb['cells'] if c.get('id')==cell_id]
                if len(cells)!=1: raise ValueError(f'Image cell missing: {name}:{cell_id}')
                pngs=[o['data']['image/png'] for o in cells[0].get('outputs',[]) if o.get('data',{}).get('image/png')]
                if ordinal>=len(pngs): raise ValueError(f'Execute notebook first: {name}:{cell_id}:{ordinal}')
                data=pngs[ordinal]; blob=base64.b64decode(''.join(data) if isinstance(data,list) else data)
                source_sha=hashlib.sha256(blob).hexdigest()
                transform=None
                if name=='03_hybridsn_baseline.ipynb' and cell_id=='legacy-16' and ordinal==0:
                    blob=redraw_historical_training(cells[0])
                    transform='Two-column redraw from saved 20 epoch logs and 4 validation logs; values unchanged; historical protocol B'
                    entry['notes']=entry.get('notes','')+' 历史曲线由保存的epoch日志重绘，数值与历史协议B不变；logged loss保留旧代码的归一化口径。'
                sha=hashlib.sha256(blob).hexdigest(); asset=assets/f'{Path(name).stem}-{cell_id}-{ordinal}-{sha[:10]}.png'; asset.write_bytes(blob)
                entry['image_asset']=str(asset)
                records.append(dict(notebook=name,cell=cell_id,ordinal=ordinal,sha256=sha,source_png_sha256=source_sha,transform=transform,path=str(asset.relative_to(ROOT/'slides')),lesson=lesson['id']))
            if entry.get('math'):
                entry['formula_asset']=formula_assets.get(entry['math'])
                if not entry['formula_asset']: raise ValueError('No reference formula asset: '+entry['math'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(lessons=LESSONS,sources=SOURCES,figures=records),ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'slides/assets/sources.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Exported',len(LESSONS),'lessons with verified notebook image cells')
if __name__=='__main__': main()
