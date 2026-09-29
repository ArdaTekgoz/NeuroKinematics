"""Confirm installed overlay artifact URL/SHA, not merely version strings."""
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[1]
pins=[]
for file in ['experiments/C1-03/dependency-pins.json','experiments/C1-03/stage2/runtime-pins.json']:
    pins.extend(json.loads((root/file).read_text(encoding='utf-8')))
results=[]
for pin in pins:
    dist=metadata.distribution(pin['name'])
    direct=json.loads(dist.read_text('direct_url.json') or '{}')
    assert dist.version==pin['version'],pin['name']
    assert Path(sys.prefix) in dist.locate_file('').resolve().parents,pin['name']+' outside overlay'
    assert direct.get('url')==pin['url'],pin['name']+' artifact URL'
    assert direct.get('archive_info',{}).get('hashes',{}).get('sha256')==pin['sha256'],pin['name']+' artifact SHA'
    results.append(dict(name=pin['name'],version=dist.version,sha256=pin['sha256'],location=str(dist.locate_file(''))))
print(json.dumps({'status':'PASS','artifacts':results},indent=2))
