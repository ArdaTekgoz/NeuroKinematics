"""Archive opaque experiment bytes; verify every member; restore without overwrites."""
import argparse,hashlib,json,shutil,subprocess,zipfile
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath

ROOT=Path(__file__).resolve().parents[1]


def digest(stream):
    h=hashlib.sha256()
    for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def sha(path):
    with path.open('rb') as f:return digest(f)


def verify(folder):
    manifest=json.loads((folder/'inventory.json').read_text(encoding='utf-8'));zpath=folder/'research-artifacts.zip'
    if sha(zpath)!=manifest['archive_sha256']:raise ValueError('archive SHA mismatch')
    with zipfile.ZipFile(zpath) as z:
        if set(z.namelist())!=set(manifest['files']):raise ValueError('member set mismatch')
        for name,entry in manifest['files'].items():
            if z.getinfo(name).file_size!=entry['bytes']:raise ValueError(name)
            with z.open(name) as f:
                if digest(f)!=entry['sha256']:raise ValueError(name)
    return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['create','verify','restore'])
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--destination',type=Path);p.add_argument('--receipt',type=Path)
    a=p.parse_args();folder=a.archive.resolve()
    if a.action=='create':
        folder.mkdir(parents=True,exist_ok=False)
        files=set(p for p in (ROOT/'data/generated').rglob('*') if p.is_file())
        files.update(p for p in (ROOT/'experiments/C1-01').rglob('*') if p.is_file() and (p.suffix=='.jsonl' or p.name.endswith('.stderr.log')))
        manifest=dict(schema='core-archive-v1',created_utc=datetime.now(timezone.utc).isoformat(),source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            scope='all data/generated plus C1-01 JSONL and stderr; no environments or secrets',sealed_data_access='opaque byte preservation only; no parse, evaluation or model selection',files={})
        with zipfile.ZipFile(folder/'research-artifacts.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            for path in sorted(files):
                name=path.relative_to(ROOT).as_posix();entry=dict(bytes=path.stat().st_size,sha256=sha(path));manifest['files'][name]=entry
                z.write(path,name)
        manifest['archive_sha256']=sha(folder/'research-artifacts.zip')
        (folder/'inventory.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    manifest=verify(folder)
    if a.action=='restore':
        if a.destination is None:raise ValueError('--destination required')
        dest=a.destination.resolve()
        targets=[]
        for name in manifest['files']:
            relative=PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe member')
            target=(dest/name).resolve()
            if not target.is_relative_to(dest) or target.exists():raise ValueError('unsafe or existing destination: '+str(target))
            targets.append((name,target))
        with zipfile.ZipFile(folder/'research-artifacts.zip') as z:
            for name,target in targets:
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(name) as src,target.open('xb') as out:shutil.copyfileobj(src,out)
                if sha(target)!=manifest['files'][name]['sha256']:raise ValueError('restored SHA mismatch: '+name)
    receipt=dict(status='PASS',action=a.action,archive=str(folder),archive_sha256=manifest['archive_sha256'],
        inventory_sha256=sha(folder/'inventory.json'),files=len(manifest['files']),raw_bytes=sum(v['bytes'] for v in manifest['files'].values()),
        archive_bytes=(folder/'research-artifacts.zip').stat().st_size,verification='all member bytes and SHA256',
        remote_archive='NOT_CONFIRMED',independent_device_backup='NOT_CONFIRMED',utc=datetime.now(timezone.utc).isoformat())
    if a.receipt:a.receipt.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
