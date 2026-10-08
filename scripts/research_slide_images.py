"""Query official/public source pages for lecture illustration candidates.

No account access, no course upload, no external image automatically included
in slides. Candidate permissions require human inspection before reuse.
"""
from pathlib import Path
import hashlib
import json
import urllib.request
from urllib.parse import urljoin,urlparse
from html.parser import HTMLParser
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
PAGES=[
 ('aviris','https://aviris.jpl.nasa.gov/html/aviris.overview.html'),
 ('nasa-media-guidelines','https://www.nasa.gov/nasa-brand-center/images-and-media/'),
 ('hyperspectral-datasets','https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes'),
 ('hybridsn-official','https://github.com/gokriznastic/HybridSN'),
 ('spectralformer-official','https://github.com/danfenghong/IEEE_TGRS_SpectralFormer'),
]


class Extract(HTMLParser):
    def __init__(self):
        super().__init__(); self.images=[]; self.title=[]; self.in_title=False
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='title': self.in_title=True
        if tag=='img' and a.get('src'): self.images.append({'src':a['src'],'alt':a.get('alt','')})
    def handle_endtag(self,tag):
        if tag=='title': self.in_title=False
    def handle_data(self,data):
        if self.in_title: self.title.append(data)


def main():
    out=ROOT/'slides/assets/research'; out.mkdir(parents=True,exist_ok=True)
    records=[]
    for key,url in PAGES:
        row=dict(id=key,url=url,accessed_utc=datetime.now(timezone.utc).isoformat(),
                 inclusion='not-approved',license='unverified')
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'HSI-Learning teaching source audit/1.0'})
            with urllib.request.urlopen(request,timeout=30) as response:
                payload=response.read(3*1024*1024+1)
                if len(payload)>3*1024*1024: raise ValueError('page exceeds 3 MiB limit')
                charset=response.headers.get_content_charset() or 'utf-8'
                row['resolved_url']=response.geturl()
            doc=payload.decode(charset,errors='replace')
            parsed=Extract(); parsed.feed(doc)
            row.update(status='fetched',sha256=hashlib.sha256(payload).hexdigest(),title=''.join(parsed.title),
                       images=[{'url':urljoin(row['resolved_url'],i['src']),'alt':i['alt'],
                                'credit':'unverified','license':'unverified'} for i in parsed.images
                               if urlparse(urljoin(url,i['src'])).scheme=='https'][:30])
            # Save source HTML as text for audit only; never execute embedded scripts.
            (out/f'{key}.html.txt').write_text(doc,encoding='utf-8')
            print(f'{key}: fetched, {len(row["images"])} image candidates; NOT approved',flush=True)
        except Exception as exc:
            row.update(status='blocked',error=str(exc)); print(f'{key}: {exc}',flush=True)
        records.append(row)
    (out/'candidates.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Candidate discovery complete. Check credits/usage terms before embedding any external image.')
    return 0 if all(r['status']=='fetched' for r in records) else 1


if __name__=='__main__': raise SystemExit(main())
