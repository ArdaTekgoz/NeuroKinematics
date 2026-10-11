"""Build shareable negative-result figures and report from frozen evidence."""
from pathlib import Path
import csv,hashlib,json,re,html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/publication/core'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'experiments/C1-06R/diagnostic9/results.json';data=json.loads(source.read_text())
    cells=data['cells'];rows=[]
    for c in cells:
        loc=c['metrics']['validation']['strata']['local']
        rows.append(dict(seed=c['seed'],arm=c['arm'],train_a=c['metrics']['train']['profile_a'],train_n=4096,
            probe_a=c['metrics']['same_root_probe']['profile_a'],probe_n=4096,
            validation_a=c['metrics']['validation']['profile_a'],validation_n=3600,
            local_position_mm=loc['position_m']['median']*1000,local_orientation_deg=loc['orientation_deg']['median'],
            local_invalid=loc['invalid'],local_n=1800))
    with (OUT/'figure-data.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    palette=['#25607f','#c86831']
    fig,axes=plt.subplots(1,3,figsize=(13,5.8));fig.subplots_adjust(top=.76,bottom=.24,wspace=.35)
    for ax,field,n,title in zip(axes,['train_a','probe_a','validation_a'],[4096,4096,3600],['Eğitim','Aynı kökte yeni yön','Görülmemiş kökler']):
        for j,arm in enumerate(['RAW','CENTERED']):
            vals=[r[field] for r in rows if r['arm']==arm]
            xs=[i+(j-.5)*.34 for i in range(3)]
            ax.bar(xs,[v/n*100 for v in vals],width=.32,color=palette[j],label=arm)
            for x,v in zip(xs,vals):ax.annotate(str(v),(x,v/n*100),xytext=(0,5),textcoords='offset points',ha='center',fontsize=10)
        ax.set_title(title);ax.set_xticks(range(3),['Seed 01','Seed 02','Seed 03']);ax.set_ylabel('Profil A başarısı (%)')
        ax.set_ylim(0,20);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(loc='upper left',fontsize=10)
    fig.suptitle('Eğitimde iyileşme, yeni köklerde başarıya dönüşmedi',fontsize=19,fontweight='bold',y=.95)
    fig.text(.5,.85,'NeuroKinematics • C1-06R Tanı 9 • Üç eşli seed, aynı veri ve 5000 güncelleme',ha='center',fontsize=12)
    fig.text(.06,.12,'Sütun üstleri başarılı örnek sayısıdır. Eğitim/yeni yön N=4096; validation N=3600.\nProfil A: ≤2 mm, ≤1° ve limit içinde q. Aynı yüzde ekseni; bütün validation B sonuçları sıfır.',fontsize=11)
    fig.text(.06,.035,'Kaynak: diagnostic9/results.json • Validation araştırma verisidir; bağımsız final değildir.',fontsize=10,color='#555555')
    for ext in ['png','svg']:fig.savefig(OUT/('generalization.'+ext),dpi=180,facecolor='white')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5.6));fig.subplots_adjust(top=.76,bottom=.25,wspace=.3)
    for ax,key,title,threshold in zip(axes,['local_position_mm','local_orientation_deg'],['Medyan konum hatası (mm)','Medyan yönelim hatası (°)'],[2,1]):
        for j,arm in enumerate(['RAW','CENTERED']):
            ax.plot(range(1,4),[r[key] for r in rows if r['arm']==arm],marker='o',color=palette[j],lw=2,label=arm)
        ax.axhline(threshold,color='#555555',ls='--',label='Profil A bileşen eşiği');ax.set_ylim(bottom=0)
        ax.set_xticks([1,2,3],['Seed 01','Seed 02','Seed 03']);ax.set_title(title);ax.grid(alpha=.2)
    axes[0].legend(fontsize=10)
    fig.suptitle('Sıfır hareket düzeldi; yeni köklerde hata arttı',fontsize=19,fontweight='bold',y=.94)
    fig.text(.5,.84,'Local validation: 1800 sorgu • RAW ve CENTERED karşılaştırması',ha='center',fontsize=12)
    fig.text(.07,.12,'Medyanlar tam paydalıdır. Ortak başarı iki pose eşiğini ve eklem limitlerini birlikte gerektirir.\nCENTERED local limit ihlali: 112 / 102 / 120; RAW: 63 / 73 / 70. Payda her seed için 1800.',fontsize=11)
    fig.text(.07,.035,'Kaynak: diagnostic9/results.json • Yeni-kök genellemesi için iyileşme kanıtı yok.',fontsize=10,color='#555555')
    for ext in ['png','svg']:fig.savefig(OUT/('local-errors.'+ext),dpi=180,facecolor='white')
    plt.close(fig)
    sources=[ROOT/'docs/research/C1-06_NEGATIVE_RESULTS.md',source,ROOT/'experiments/C1-06R/diagnostic9/audit.json']
    for name in ['round1-analysis']+[f'diagnostic{i}' for i in range(2,10)]:
        sources.extend(p for p in (ROOT/'experiments/C1-06R'/name).rglob('*') if p.suffix in {'.md','.json'})
    sources.extend([ROOT/'experiments/C1-06/stage2/RESULTS.md',ROOT/'experiments/C1-06/stage2/final-001/acceptance.json'])
    sources.extend((ROOT/'docs/adr').glob('*.md'))
    sources.extend([ROOT/'docs/raporlar/02_Core_v1_0_r1.md',ROOT/'experiments/C1-07/preparation/C1-06R-closure.json',ROOT/'experiments/C1-07/preparation/G1_READINESS.md'])
    (OUT/'evidence-index.json').write_text(json.dumps(dict(sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        figures_source=source.relative_to(ROOT).as_posix(),figure_scope='diagnostic9, all six cells; no pooled independent N',
        report_revision='r1',old_final_raw='NOT_READ',linkedin_published=False),indent=2)+'\n',encoding='utf-8')
    pdfmetrics.registerFont(TTFont('Body','C:/Windows/Fonts/arial.ttf'));pdfmetrics.registerFont(TTFont('Strong','C:/Windows/Fonts/arialbd.ttf'))
    styles=getSampleStyleSheet()
    for key in styles.byName:
        styles[key].fontName='Body'
    styles['Title'].fontName='Strong';styles['Title'].fontSize=21;styles['Title'].leading=26
    styles['Heading1'].fontName='Strong';styles['Heading1'].fontSize=14;styles['Heading1'].leading=18
    styles['BodyText'].fontSize=10;styles['BodyText'].leading=14;styles['BodyText'].spaceAfter=7
    styles.add(ParagraphStyle('Cell',fontName='Body',fontSize=8,leading=11))
    lines=sources[0].read_text(encoding='utf-8').splitlines();story=[];buffer=[];table=[]
    def para():
        if buffer:story.append(Paragraph(html.escape(' '.join(buffer)),styles['BodyText']));buffer.clear()
    def tab():
        if table:
            vals=[[Paragraph(html.escape(v.strip()),styles['Cell']) for v in row.strip('|').split('|')] for row in table if not re.match(r'^\|[- :|]+\|$',row)]
            widths=[105,170,220] if len(vals[0])==3 else [70,140,140,145]
            t=Table(vals,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef2')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#cccccc')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
            story.extend([t,Spacer(1,10)]);table.clear()
    for line in lines:
        if line.startswith('|'):para();table.append(line);continue
        tab()
        if line.startswith('# '):para();story.append(Paragraph(html.escape(line[2:]),styles['Title']))
        elif line.startswith('## '):para();story.append(Paragraph(html.escape(line[3:]),styles['Heading1']))
        elif not line:para()
        elif line.startswith('- '):para();story.append(Paragraph(html.escape(line[2:]),styles['BodyText']))
        else:buffer.append(line)
    para();tab();story.append(Paragraph('Kaynak veriden üretilen grafikler',styles['Heading1']))
    for name in ['generalization.png','local-errors.png']:story.extend([Image(str(OUT/name),width=495,height=230),Spacer(1,18)])
    def footer(canvas,doc):
        canvas.setFont('Body',8);canvas.drawString(42,25,'NeuroKinematics • Core negatif sonuç raporu • r1 • 11 Ekim 2026');canvas.drawRightString(552,25,str(doc.page))
    SimpleDocTemplate(str(OUT/'Core_Negatif_Sonuc_Raporu.pdf'),pagesize=(595.28,841.89),rightMargin=45,leftMargin=45,topMargin=42,bottomMargin=44).build(story,onFirstPage=footer,onLaterPages=footer)
    print('Built report, two figures and evidence index:',OUT)


if __name__=='__main__':main()
