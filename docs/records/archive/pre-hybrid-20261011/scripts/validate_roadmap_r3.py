"""Validate the new plan DAG, active links and unchanged source evidence."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import sys
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/PLAN-20261011-ROADMAP'
def read(path): return json.loads((ROOT/path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024), b''): h.update(part)
    return h.hexdigest()

def main():
    errors=[]; checks=[]
    def check(ok,name):
        checks.append({'check':name,'passed':bool(ok)})
        if not ok: errors.append(name)
    data=read('docs/roadmaps/ROADMAP_TASKS_r3.json');tasks=data['tasks']
    ids=[t['id'] for t in tasks];by={t['id']:t for t in tasks}
    check(len(ids)==len(set(ids))==39,'39 unique future task IDs')
    complete=set(data['historical_prerequisites']);order=[];remaining=set(ids)
    while remaining:
        ready=[i for i in ids if i in remaining and set(by[i]['dependencies']) <= complete]
        if not ready: break
        for i in ready: remaining.remove(i);complete.add(i);order.append(i)
    check(not remaining,'DAG has no cycles or missing dependencies')
    check([i for i in ids if set(by[i]['dependencies'])<=set(data['historical_prerequisites'])]==['H2-01'], 'Only H2-01 is initially ready')
    for t in tasks:
        p=ROOT/f"docs/tasks/{t['id']}.md"
        check(p.is_file(),f"Task file exists: {t['id']}")
        text=p.read_text(encoding='utf-8') if p.exists() else ''
        check(all(v in text for v in t['requirements']+t['tests']+t['dependencies']),f"Trace fields match: {t['id']}")
        check('NOT_STARTED' in text and 'NOT_RUN' in text, f"No invented implementation result: {t['id']}")
        check(t['id'] in (ROOT/'docs/TRACEABILITY.md').read_text(encoding='utf-8'), f"Trace row present: {t['id']}")
    hours=[t['hours'] for t in tasks if t['hours']]
    check([sum(h[i] for h in hours) for i in range(2)]==[436,708], 'Effort total equals 436–708 hours')
    def ancestors(i):
        out=set(by[i]['dependencies'])
        for d in list(out):
            if d in by: out |= ancestors(d)
        return out
    check({'PE-05A','PE-06','PE-07B','PE-01'}<=ancestors('PE-04'),'Physical execution follows calibration/HIL/interlocks/twin')
    check('MR-06' in ancestors('S3-02'),'Generic onboarding before GUI')
    check('TR-04' in ancestors('MP-01') and 'MP-05' in ancestors('S3-03'),'TR before MP; integrated Studio after MP')
    for item in read('experiments/PLAN-20261011-ROADMAP/source-inventory.json'):
        check(sha(ROOT/item['path'])==item['sha256'], 'Unchanged source PDF: '+item['name'])
    for item in read('experiments/PLAN-20261011-ROADMAP/pre-revision-manifest.json'):
        check(sha(ROOT/item['snapshot'])==item['sha256'], 'Exact pre-revision snapshot: '+item['original'])
    for manifest in ['experiments/C1-06R/training-freeze.json','experiments/C1-06S/final-freeze.json']:
        for name,digest in read(manifest)['files'].items():
            check(sha(ROOT/name)==digest,'Unchanged frozen file: '+name)
    for item in read('experiments/C1-06S/evidence-manifest.json')['files']:
        check(sha(ROOT/item['path'])==item['sha256'],'Unchanged geometric evidence: '+item['path'])
    for name,base in [('experiments/C1-07/SHA256SUMS','experiments/C1-07/'),('experiments/C1-07/DELIVERY_SHA256SUMS','')]:
        for line in (ROOT/name).read_text(encoding='utf-8').splitlines():
            if line.strip():
                digest,path=line.split(maxsplit=1);path=base+path.strip().lstrip('*')
                check(sha(ROOT/path)==digest,'Unchanged Core delivery: '+path)
    sources=[ROOT/f'docs/tasks/{i}.md' for i in ids]
    sources += [ROOT/'docs/roadmaps'/name for name in ['MASTER_ROADMAP.md','H2_Hybrid.md','H2_REVISION_SPEC.md','MR_MultiRobot.md','TR_Trajectory.md','MP_MotionPlanning.md','S3_Studio.md','PE_PhysicalExecution.md','SOURCE_MAPPING_r3.md']]
    sources += [ROOT/name for name in ['docs/MASTER_ROADMAP.md','CURRENT_ROADMAP.md','docs/adr/ADR-027-roadmap-revision-and-neural-role.md','experiments/PLAN-20261011-ROADMAP/C1_READINESS_AUDIT.md','experiments/PLAN-20261011-ROADMAP/RUN_REPORT.md']]
    link_count=0
    for p in sources:
        check(p.is_file(),'Active document exists: '+p.relative_to(ROOT).as_posix())
        if not p.is_file(): continue
        for m in re.finditer(r'\[[^\]\n]+\]\((<[^>]+>|[^)\n]+)\)',p.read_text(encoding='utf-8')):
            target=m.group(1).strip('<>');target=unquote(target.split('#')[0])
            if not target or re.match(r'[a-z]+://',target): continue
            link_count+=1
            check((p.parent/target).resolve().exists(),f"Active link: {p.relative_to(ROOT)} -> {target}")
    result=dict(timestamp=datetime.now(timezone.utc).isoformat(),passed=not errors,checks=len(checks),
                task_count=len(tasks),topological_order=order,active_links=link_count,errors=errors,details=checks)
    destination=BASE/'plan-validation.json'
    # Validation is rerunnable; retain each failed/successful audit rather than overwrite.
    if destination.exists():
        n=2
        while (BASE/f'plan-validation-r{n}.json').exists(): n+=1
        destination=BASE/f'plan-validation-r{n}.json'
    destination.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['details','topological_order']},ensure_ascii=False))
    sys.exit(0 if not errors else 1)

if __name__=='__main__':main()
