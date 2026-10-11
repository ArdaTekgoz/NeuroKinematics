"""Package only reviewed social assets and record the evidence sources."""
import hashlib
import json
from pathlib import Path
import zipfile

from PIL import Image
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"output/linkedin/neurokinematics-core-progress"
SOURCE = ROOT/"docs/publication/core-progress-20261011"
PDF = ROOT/"output/pdf/NeuroKinematics_Foundations_Core.pdf"
text = (SOURCE/"LINKEDIN.md").read_text(encoding="utf-8")
caption = text.split("## Paylaşım metni\n\n", 1)[1].split("\n## Görsel sırası", 1)[0].strip()+"\n"
(OUT/"LinkedIn_Metni.txt").write_text(caption, encoding="utf-8")
alt = text.split("## Görsel sırası ve alternatif metinler\n\n", 1)[1].split("\n## Paylaşım notları", 1)[0].strip()+"\n"
(OUT/"Gorsel_Aciklamalari.txt").write_text(alt, encoding="utf-8")
pdf = pymupdf.open(PDF)
assert len(pdf) == 6
extracted = "\n".join(page.get_text() for page in pdf)
for expected in ("İlk iki faz", "tamamlandı.", "12.000 / 12.000", "315 test geçti", "Sırada Hybrid."):
    assert expected in extracted, expected
assert "\ufffd" not in extracted
images = sorted(OUT.glob("[0-9][0-9]-neurokinematics.png"))
assert len(images) == 6
for image in images:
    assert Image.open(image).size == (1080, 1350)
poppler_images = sorted((ROOT/"tmp/pdfs/core-progress").glob("page-*.png"))
assert len(poppler_images) == 6
contact = Image.new("RGB", (1080, 900), "#0B1728")
for i, path in enumerate(poppler_images):
    image = Image.open(path)
    assert image.size == (1080, 1350)
    image.thumbnail((360, 450))
    contact.paste(image, ((i % 3)*360, (i//3)*450))
contact.save(ROOT/"tmp/pdfs/core-progress/poppler-overview.png")
archive_path = OUT/"NeuroKinematics_LinkedIn_Gorseller.zip"
with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in images+[OUT/"LinkedIn_Metni.txt", OUT/"Gorsel_Aciklamalari.txt"]:
        archive.write(path, arcname=path.name)

def record(path):
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())

sources = ["experiments/F0-06/G0_DECISION.md", "experiments/C1-07/G1_DECISION.md",
    "experiments/C1-07/HYBRID_HANDOFF.json", "experiments/C1-06S/validation.json",
    "experiments/C1-06S/final1.json", "experiments/C1-06S/acceptance.json",
    "experiments/C1-06S/RECOVERY_DECISION.md", "experiments/C1-06S/final1-inputs.json",
    "docs/roadmaps/H2_Hybrid.md", "docs/raporlar/03_Hybrid_v2_0_r1.md",
    "docs/adr/ADR-026-geometric-core-recovery.md"]
manifest = dict(status="GENERATED_AND_STRUCTURALLY_VERIFIED", pdf_pages=6,
    images=6, image_dimensions=[1080,1350], caption_characters=len(caption.strip()),
    pdf_text_check="PASS", bounds_check="PASS", poppler_render="6_PAGES",
    neural_retrained=False, roadmap_modified=False, linkedin_published=False,
    source_files=[record(ROOT/name) for name in sources],
    output_files=[record(p) for p in images+[PDF, archive_path, OUT/"LinkedIn_Metni.txt", OUT/"Gorsel_Aciklamalari.txt"]])
(SOURCE/"artifact-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(json.dumps({k: v for k, v in manifest.items() if k not in ("source_files", "output_files")}, ensure_ascii=True))
