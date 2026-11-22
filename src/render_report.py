"""Render every final PDF page and contact sheets for visual inspection."""
from pathlib import Path
import shutil
import subprocess
import fitz
from PIL import Image,ImageOps,ImageDraw
from common import ROOT,save_json

def main():
    path=ROOT/'PA3_25280025_Muhammad_Sami_Ul_Basit.pdf'
    out=ROOT/'tmp'/'pdfs';out.mkdir(parents=True,exist_ok=True)
    doc=fitz.open(path)
    poppler=shutil.which('pdftoppm')
    rendered=[]
    if poppler:
        subprocess.run([poppler,'-r','100','-png',str(path),str(out/'report')],check=True)
        rendered=sorted(out.glob('report-*.png'))[:len(doc)]
        assert len(rendered)==len(doc)
    records=[]
    thumbs=[]
    for i,page in enumerate(doc):
        if rendered:
            png=rendered[i]
        else:
            pix=page.get_pixmap(matrix=fitz.Matrix(1.4,1.4),alpha=False)
            png=out/f'page_{i+1:02d}.png';pix.save(png)
        text=page.get_text(sort=True)
        blocks=page.get_text('blocks')
        outside=[list(b[:4]) for b in blocks if b[0]<10 or b[1]<10 or b[2]>page.rect.width-10 or b[3]>page.rect.height-10]
        assert not outside,(i+1,outside)
        assert '\ufffd' not in text
        records.append(dict(page=i+1,first_line=text.splitlines()[0],characters=len(text),outside_bounds=outside))
        thumb=Image.open(png).convert('RGB');thumb.thumbnail((595,842))
        framed=Image.new('RGB',(615,877),'#d8e1e7')
        framed.paste(thumb,((615-thumb.width)//2,25))
        ImageDraw.Draw(framed).text((12,7),f'Page {i+1}',fill='black')
        thumbs.append(framed)
    for offset in range(0,len(thumbs),4):
        sheet=Image.new('RGB',(1230,1754),'#d8e1e7')
        for j,thumb in enumerate(thumbs[offset:offset+4]):
            sheet.paste(thumb,((j%2)*615,(j//2)*877))
        sheet.save(out/f'contact_{offset//4+1}.png')
    save_json('pdf_checks.json',dict(pages=len(doc),page_records=records,rendered_all_pages=True,renderer='Poppler' if poppler else 'PyMuPDF'))
    print('Rendered',len(doc),'pages to',out)
    print('\n'.join(f'{r["page"]}: {r["first_line"]}' for r in records))

if __name__=='__main__':
    main()
