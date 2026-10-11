"""Evidence-backed, six-page Turkish social graphic set; no solver changes."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/linkedin/neurokinematics-core-progress"
PDF = ROOT / "output/pdf/NeuroKinematics_Foundations_Core.pdf"
SOURCE = ROOT / "docs/publication/core-progress-20261011"
BG, PANEL, INK, MUTED = "#0B1728", "#14243A", "#F4F7FB", "#AFC0D5"
CYAN, GREEN, AMBER = "#5CDCE8", "#B4ED84", "#F4BB73"


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def evidence():
    validation = read("experiments/C1-06S/validation.json")
    final = read("experiments/C1-06S/final1.json")
    handoff = read("experiments/C1-07/HYBRID_HANDOFF.json")
    models = [x for x in handoff["candidates"] if x["family"] == "FK_TANH"]
    assert len(models) == 3
    assert all(x["validation"]["profile_a"] == 0 and x["validation"]["profile_b"] == 0
               and x["validation"]["n"] == 3600 for x in models)
    assert validation["groups"]["all"]["profile_a"] == 3600
    assert validation["groups"]["missing_teacher"]["n"] == 351
    assert final["unique_queries"] == 12000 and final["measurements"] == 60000
    assert final["groups"]["all"]["profile_a"] == final["groups"]["all"]["profile_b"] == 12000
    return validation, final


def charts():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import numpy as np
    evidence()
    OUT.mkdir(parents=True, exist_ok=True)
    font = FontProperties(fname="C:/Windows/Fonts/segoeui.ttf")
    plt.rcParams.update({"font.family": font.get_name(), "font.size": 18,
                         "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": INK, "axes.edgecolor": MUTED, "savefig.transparent": True})
    target_path = ROOT / "data/generated/C1-06S/final1-inputs/requests.json"
    meta = read("experiments/C1-06S/final1-inputs.json")
    assert hashlib.sha256(target_path.read_bytes()).hexdigest() == meta["requests_sha256"]
    positions = np.array([v["position_m"] for v in json.loads(target_path.read_text())])
    fig, ax = plt.subplots(figsize=(11, 5.0), dpi=180)
    ax.scatter(positions[:, 0], positions[:, 2], s=2, c=CYAN, alpha=.18, rasterized=True)
    ax.set(xlabel="X (m)", ylabel="Z (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(alpha=.10, color=MUTED)
    fig.tight_layout(pad=.3)
    fig.savefig(OUT / "targets-projection.png")
    plt.close(fig)

    def bars(filename, labels, values, notes, colors, height):
        fig, ax = plt.subplots(figsize=(11.3, height), dpi=180)
        ys = np.arange(len(labels))[::-1]
        ax.barh(ys, [100]*len(labels), height=.38, color=PANEL)
        ax.barh(ys, values, height=.38, color=colors)
        for y, value, note in zip(ys, values, notes):
            ax.text(100, y+.28, note, ha="right", va="bottom", fontsize=20, fontweight="bold", color=INK)
            if value == 0:
                ax.plot(0, y, "o", color=AMBER, markersize=8, clip_on=False)
        ax.set_yticks(ys, labels, fontsize=17)
        ax.set_xlim(0, 100)
        ax.set_ylim(-.5, len(labels)-.3)
        ax.set_xticks([0, 20, 40, 60, 80, 100], ["%0", "%20", "%40", "%60", "%80", "%100"], fontsize=15)
        ax.tick_params(axis="y", length=0, pad=14)
        ax.tick_params(axis="x", length=0, pad=10)
        ax.spines[["right", "top", "left"]].set_visible(False)
        ax.spines["bottom"].set_alpha(.3)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=.12, color=MUTED)
        fig.subplots_adjust(left=.28, right=.98, top=.97, bottom=.18)
        fig.savefig(OUT / filename)
        plt.close(fig)
    bars("validation-comparison.png", ["Neural adaylar\nFK_TANH / 3 seed", "Geometri\nbileşeni"], [0, 100],
         ["Her modelde 0 / 3.600", "3.600 / 3.600"], [AMBER, GREEN], 4.3)
    bars("final-results.png", ["Ana küme", "Eklem sınırı", "Tekillik"], [100, 100, 100],
         ["10.000 / 10.000", "1.000 / 1.000", "1.000 / 1.000"], [CYAN, GREEN, GREEN], 4.8)
    with (SOURCE / "figure-data.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["set", "component", "group", "profile_a", "profile_b", "n", "independent_targets_note"])
        writer.writerow(["validation", "FK_TANH_each_of_3", "all", 0, 0, 3600, "local/wide share roots"])
        writer.writerow(["validation", "analytic-kr6-r1", "all", 3600, 3600, 3600, "same queries as neural"])
        for group, n in (("main", 10000), ("boundary", 1000), ("singularity", 1000)):
            writer.writerow(["new_final", "analytic-kr6-r1", group, n, n, n, "five timing passes are repeats"])
    print("Charts and exact figure data saved")


def document():
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.colors import HexColor
    from reportlab.lib.utils import ImageReader
    from PIL import Image, ImageOps, ImageDraw
    import pymupdf
    evidence()
    PDF.parent.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(TTFont("Segoe", "C:/Windows/Fonts/segoeui.ttf"))
    pdfmetrics.registerFont(TTFont("SegoeBold", "C:/Windows/Fonts/segoeuib.ttf"))
    c = canvas.Canvas(str(PDF), pagesize=(1080, 1350), pageCompression=1)
    c.setTitle("NeuroKinematics | Foundations ve Core")
    c.setAuthor("Arda Tekgöz")
    text_boxes = []

    def box(x, y, w, h, fill=PANEL, radius=20):
        c.setFillColor(HexColor(fill))
        c.roundRect(x, 1350-y-h, w, h, radius, fill=1, stroke=0)

    def txt(x, y, value, size=30, bold=False, color=INK):
        face = "SegoeBold" if bold else "Segoe"
        width = pdfmetrics.stringWidth(value, face, size)
        if x < 0 or x+width > 1012 or y-size < 0 or y+10 > 1350:
            raise ValueError(f"Text overflow: {value} width={width}")
        text_boxes.append(dict(page=page, x=x, y=y, width=width, size=size, text=value))
        c.setFillColor(HexColor(color)); c.setFont(face, size)
        c.drawString(x, 1350-y, value)

    def lines(x, y, values, size=30, bold=False, color=INK, gap=None):
        gap = size*1.35 if gap is None else gap
        for i, value in enumerate(values):
            txt(x, y+i*gap, value, size, bold, color)

    def img(name, x, y, w, h):
        c.drawImage(ImageReader(str(OUT/name)), x, 1350-y-h, width=w, height=h, mask="auto", preserveAspectRatio=True, anchor="c")

    def begin(number, category):
        nonlocal page
        page = number
        c.setFillColor(HexColor(BG)); c.rect(0, 0, 1080, 1350, fill=1, stroke=0)
        box(72, 61, 9, 29, CYAN, 3)
        txt(99, 85, "NEUROKINEMATICS", 24, True)
        txt(72, 148, category.upper(), 23, False, CYAN)
        c.setStrokeColor(HexColor(PANEL)); c.setLineWidth(2)
        c.line(72, 100, 1008, 100)
        txt(72, 1290, "Arda Tekgöz  /  11 Ekim 2026", 23, False, MUTED)
        txt(932, 1290, f"{number:02d} / 06", 21, False, MUTED)

    page = 0
    begin(1, "Proje günlüğü")
    lines(72, 259, ["İlk iki faz", "tamamlandı."], 88, True, gap=102)
    lines(72, 433, ["Planla. Ölç. Öğren. Geliştir."], 35, False, MUTED)
    img("targets-projection.png", 78, 490, 920, 400)
    txt(164, 925, "12.000 yeni hedefin TCP konum izdüşümü", 27, False, MUTED)
    box(72, 977, 936, 93)
    txt(99, 1016, "01  FOUNDATIONS", 26, True, CYAN)
    txt(99, 1050, "Robot, matematik ve veri doğrulaması: G0 kabulü", 26)
    box(72, 1090, 936, 119)
    txt(99, 1129, "02  CORE", 26, True, GREEN)
    txt(99, 1163, "G1 araştırma kabulü + geometri çözümünde başarı", 26)

    c.showPage(); begin(2, "Faz 01 / Foundations")
    lines(72, 248, ["Önce matematiği", "doğruladım."], 65, True, gap=79)
    txt(72, 388, "Öğrenmenin üzerinde yükseleceği temel.", 30, False, MUTED)
    items = [
        ("01", "Robot ve koordinat sözleşmesi", ["Model, birimler, eklem sırası ve TCP sabitlendi."]),
        ("02", "İki bağımsız ileri kinematik", ["Aynı eklem konfigürasyonlarının pozları", "referans ve bağımsız uygulamada karşılaştırıldı."]),
        ("03", "Jacobian ve türev denetimi", ["Analitik sonuçlar ve sonlu farklar karşılaştırıldı;", "yanlış hesapları yakalayan kontroller kuruldu."]),
        ("04", "Veri ve sayısal referans", ["Deterministik üretim, ayrık veri bölümleri", "ve DLS baseline ölçüm hattı hazırlandı."])]
    for i, (idx, title, body) in enumerate(items):
        y = 440+i*177
        box(72, y, 936, 155)
        txt(98, y+61, idx, 42, True, CYAN)
        txt(190, y+48, title, 32, True)
        lines(190, y+90, body, 27, False, MUTED, gap=36)
    txt(72, 1201, "10.000 konfigürasyonla FK eşliği kontrol edildi.", 29, True, GREEN)

    c.showPage(); begin(3, "Faz 02 / Core")
    lines(72, 248, ["Modelleri ve", "varsayımları sınadım."], 62, True, gap=78)
    for x, number, title, sub in [(72, "24.000", "durum-hedef çifti", "Yerel ve geniş başlangıçlar"),
                                  (555, "600.000", "baseline ölçümü", "Beş çözücü varyantı")]:
        box(x, 389, 453, 215)
        txt(x+25, 465, number, 65, True, CYAN)
        txt(x+25, 523, title, 30, True)
        txt(x+25, 566, sub, 24, False, MUTED)
    steps = [("Diferansiyellenebilir FK", "İleri kinematik ve gradyan eşliği"),
             ("Neural modeller ve fizik kayıpları", "Kontrollü ablasyonlar, eşli eğitim seed'leri"),
             ("Bağımsız değerlendirme", "Hata analizi, model kartı ve tekrar üretim")]
    for i, (title, sub) in enumerate(steps):
        y = 676+i*98
        txt(72, y, title, 32, True)
        txt(72, y+39, sub, 27, False, MUTED)
    box(72, 987, 936, 209)
    txt(99, 1031, "DENEYLERİN GÖSTERDİĞİ", 24, True, AMBER)
    lines(99, 1084, ["İlk neural modeller doğrudan IK için", "gereken hassasiyeti sağlayamadı.", "Ek incelemeler mimariyi geliştirmemizi sağladı."], 29, gap=39)

    c.showPage(); begin(4, "Dönüm noktası / mimari değişimi")
    lines(72, 248, ["Aynı sorgular.", "Yeni çözüm mimarisi."], 62, True, gap=78)
    lines(72, 390, ["Aynı 3.600 validation sorgusu ve aynı kabul eşikleri.", "Profil A ve Profil B sonuçları eşit."], 29, False, MUTED, gap=40)
    img("validation-comparison.png", 65, 484, 948, 375)
    box(72, 891, 936, 148)
    txt(99, 936, "DEĞİŞEN BİLEŞEN", 24, True, GREEN)
    lines(99, 980, ["Robot geometrisinden çözüm dalları + bağımsız FK.", "Bu sonuç yeni bir neural model eğitimi değildir."], 28, gap=38)
    txt(72, 1117, "351", 63, True, CYAN)
    lines(216, 1085, ["Eski öğretmenin etiket üretemediği", "wide sorgular da çözüldü."], 29, False, MUTED, gap=40)
    txt(72, 1210, "Validation araştırma kümesidir; yeni final sonraki sayfada.", 25, False, MUTED)

    c.showPage(); begin(5, "Yeni final / kod sabitlendikten sonra")
    lines(72, 248, ["Yeni hedeflerde", "bağımsız doğrulama."], 62, True, gap=78)
    txt(72, 441, "12.000 / 12.000", 81, True, GREEN)
    txt(72, 493, "Test edilen sorgularda kinematik başarı", 30, False, MUTED)
    img("final-results.png", 58, 548, 963, 409)
    box(72, 970, 936, 102)
    txt(99, 1012, "Profil A: 2 mm / 1°      Profil B: 1 mm / 0,5°", 28, True)
    txt(99, 1050, "İki profil de sağlandı; eklem limitleri korundu.", 27, False, MUTED)
    txt(72, 1133, "60.000 ölçüm", 34, True, CYAN)
    txt(557, 1133, "315 test geçti", 34, True, CYAN)
    lines(72, 1184, ["12.000 hedef × 5 geçiş. Tek robot: KR 6 R900 sixx.", "Fiziksel robot ve çarpışma güvenliği doğrulanmadı."], 25, False, MUTED, gap=35)

    c.showPage(); begin(6, "Bir sonraki faz / Hybrid")
    txt(72, 257, "Sırada Hybrid.", 76, True)
    lines(72, 337, ["Doğrulanmış çözümü", "bütçeli bir servise dönüştürmek."], 38, False, MUTED, gap=49)
    stages = [("H2-01", "Kabul ve hata politikası"), ("H2-02", "Bütçeli çözüm ve fallback"),
              ("H2-03", "Neural katkının adil kıyası"), ("H2-04", "Yörünge ve çarpışma kontrolü"),
              ("H2-05", "Neural bileşende ONNX eşliği"), ("H2-06", "Servis paketi ve G2 kapanışı")]
    for i, (number, title) in enumerate(stages):
        y = 461+i*91
        box(72, y, 936, 73)
        txt(96, y+47, number, 27, True, CYAN)
        txt(221, y+47, title, 31, True)
    lines(72, 1079, ["Ana sıra ve kabul eşikleri korunuyor.", "Geometri entegrasyonu ve kıyas tasarımı güncellenecek."], 28, False, MUTED, gap=40)
    txt(72, 1210, "İlerleme: ölçmek, öğrenmek ve yöntemi geliştirmek.", 30, True, GREEN)
    c.showPage(); c.save()
    document = pymupdf.open(PDF)
    cards = []
    for i, p in enumerate(document):
        target = OUT / f"{i+1:02d}-neurokinematics.png"
        p.get_pixmap(matrix=pymupdf.Matrix(1, 1), alpha=False).save(target)
        assert Image.open(target).size == (1080, 1350)
        cards.append(target)
    contact = Image.new("RGB", (1080, 900), BG)
    for i, path in enumerate(cards):
        thumb = ImageOps.contain(Image.open(path), (360, 450))
        contact.paste(thumb, ((i % 3)*360, (i//3)*450))
    contact.save(OUT / "overview.png")
    (SOURCE / "layout-check.json").write_text(json.dumps(dict(pages=len(document), png_dimensions=[1080, 1350],
        text_bounds="PASS", text_boxes=text_boxes), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("PDF and six rendered PNGs saved:", PDF)


if __name__ == "__main__":
    if sys.argv[1:] == ["charts"]:
        charts()
    elif sys.argv[1:] == ["document"]:
        document()
    else:
        raise SystemExit("use charts or document")
