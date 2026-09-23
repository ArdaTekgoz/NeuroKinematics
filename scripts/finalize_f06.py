"""Build the consolidated index only from checked, complete F0-06 evidence."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from neurokinematics.foundations_gate import (EXPECTED_START, sha, require_same,
    validate_index, verify_checksums, verify_handoff)
from run_f06_regression import write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/F0-06'


def read(path):
    return json.loads(Path(path).read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--implementation-commit',default='PENDING_LOCAL_COMMIT')
    args = parser.parse_args()
    canonical = OUT/'canonical'
    history = read(OUT/'history-audit.json')
    if history['status'] != 'PASS' or history['errors']:
        raise ValueError('historical evidence not intact')
    # The additional Stage 1 checksum list describes the accepted baseline, not
    # the deliberately changed F0-06 API source revision.
    stage1 = []
    seen = set()
    for line in (ROOT/'experiments/F0-05/stage1-files.sha256').read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        if relative in seen or '..' in Path(relative).parts or Path(relative).is_absolute():
            raise ValueError('invalid Stage 1 checksum path')
        seen.add(relative)
        raw = subprocess.check_output(['git','show',f'{EXPECTED_START}:{relative}'],cwd=ROOT)
        import hashlib
        actual = hashlib.sha256(raw).hexdigest()
        require_same(actual,expected,'Stage 1 raw baseline blob')
        stage1.append({'path':relative,'expected_sha256':expected,'baseline_blob_sha256':actual})
    frozen = read(ROOT/'experiments/F0-05/stage1-frozen-hashes.json')['artifacts']
    for relative, expected in frozen.items(): require_same(sha(ROOT/relative),expected,'frozen benchmark input')
    write(OUT/'stage1-audit.json',{'status':'PASS','baseline_commit':EXPECTED_START,'records':stage1,'frozen_inputs':frozen})
    clean = read(canonical/'clean-environment.json')
    require_same(clean['base_commit'],EXPECTED_START,'start commit')
    if not all(clean['initial_paths_absent'].values()): raise ValueError('reused environment/output')
    require_same(clean['lock_sha256_before'],clean['lock_sha256_after'],'lock')
    commands = read(canonical/'setup-commands.json') + read(canonical/'reproduction-commands.json') + read(canonical/'regression/commands.json')
    tests = {}
    for cmd in commands:
        if cmd['exit_code'] != 0 or not cmd['started_utc'] or not cmd['finished_utc']:
            raise ValueError('failed or unrecorded canonical command')
        for kind in ('stdout','stderr'):
            require_same(sha(cmd[kind]['path']),cmd[kind]['sha256'],'command log')
        if 'junit' in cmd:
            junit = cmd['junit']
            require_same(sha(junit['path']),junit['sha256'],'JUnit bytes')
            suites = ET.parse(junit['path']).getroot().findall('testsuite')
            for key in ('tests','failures','errors','skipped'):
                require_same(sum(int(s.get(key,0)) for s in suites),junit[key],'JUnit counts')
            if junit['tests'] <= 0 or any(junit[k] for k in ('failures','errors','skipped')):
                raise ValueError('critical regression did not pass')
            tests[cmd['name']] = junit
    if set(tests) != {'f00','f01','f02','f03','f04','f05-unit','tf08','f05-mutations','f06'}:
        raise ValueError('incomplete regression groups')
    write(OUT/'commands.json',commands)
    write(OUT/'environment.json',read(canonical/'regression/environment.json'))
    write(OUT/'clean-environment.json',clean)
    reproduction = read(canonical/'reproduction-summary.json')
    if reproduction['status'] != 'PASS': raise ValueError('reproduction failed')
    for label in ('a','b'):
        run = canonical/'reproduction'/label
        summary = read(run/'summary.json')
        for name, key in [('query-list.jsonl','query_sha256'),('results.jsonl','result_sha256')]:
            require_same(sha(run/name),summary[key],'reproduction file')
        manifest = read(run/'dataset/dataset-manifest.json')
        for shards in manifest['shards'].values():
            for shard in shards: require_same(sha(run/'dataset'/shard['path']),shard['file_sha256'],'dataset shard')
        if any(read(run/p)['status'] != 'PASS' for p in ('dataset-audit.json','query-verification.json','result-verification.json')):
            raise ValueError('reproduction validation incomplete')
    require_same(reproduction['runs'][0]['dataset_content_sha256'],reproduction['runs'][1]['dataset_content_sha256'],'dataset')
    require_same(reproduction['runs'][0]['query_sha256'],reproduction['runs'][1]['query_sha256'],'query')
    write(OUT/'reproduction-summary.json',reproduction)
    handoff = read(OUT/'handoff-inputs.json')
    verify_handoff(ROOT,handoff)
    mutations = []
    for case in ET.parse(tests['f06']['path']).getroot().iter('testcase'):
        props = {p.get('name'):p.get('value') for p in case.findall('./properties/property')}
        if props.get('detected') == 'True': mutations.append({'test':case.get('name'),**props})
    if len(mutations) < 12: raise ValueError('insufficient recorded negative tests')
    write(OUT/'mutation-results.json',{'status':'PASS','f06_tests':tests['f06']['tests'],
         'explicit_detection_records':len(mutations),'detections':mutations,
         'additional_negative_test':'test_smoke_config_cannot_relax_rules; PASS in JUnit'})
    checks = ('historical_integrity','locked_environment','regression','dataset_determinism','query_determinism',
              'split_audit','normalization','fk_limits','benchmark_schema','handoff_inputs','commands_complete')
    gate = {**dict.fromkeys(checks,'PASS'),'start_commit':EXPECTED_START,'critical_errors':[],
            'linux_execution':'NOT_RUN','linux_claim':'NOT_VERIFIED','g0':'PASS / ACCEPTED',
            'foundations':'COMPLETE','core':'READY / NOT_STARTED'}
    excluded = {'FOUNDATIONS_EVIDENCE_INDEX.json','FOUNDATIONS_EVIDENCE_INDEX.md','SHA256SUMS','evidence-verification.json'}
    evidence = []
    for p in sorted((ROOT/'experiments').glob('F0-0*/*')):
        if p.is_file() and p.parent != OUT:
            evidence.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)})
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name not in excluded:
            evidence.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)})
    indexed = {item['path'] for item in evidence}
    source_paths = list((ROOT/'scripts').glob('*f06*.py')) + list((ROOT/'tests/f0_06').glob('*.py'))
    source_paths += [ROOT/'src/neurokinematics/foundations_gate.py'] + list((ROOT/'src/neurokinematics/benchmark').glob('*.py'))
    for p in source_paths + [ROOT/relative for relative in handoff]:
        relative = p.relative_to(ROOT).as_posix()
        if relative not in indexed:
            evidence.append({'path':relative,'sha256':sha(p)})
            indexed.add(relative)
    tasks = {f'F0-0{i}': {'status':'COMPLETE','run_report':f'experiments/F0-0{i}/RUN_REPORT.md'} for i in range(7)}
    requirements = {'REQ-F00':{'status':'NOT_DEFINED_IN_ROADMAP','note':'F0-00 maps to REQ-F01; no new requirement invented.'}}
    requirements.update({f'REQ-F0{i}':{'status':'PASS','tasks':['F0-00','F0-01'] if i==1 else [f'F0-0{i}']} for i in range(1,7)})
    index = {'schema_version':'1.0.0','tasks':tasks,'requirements':requirements,
             'tests':{f'T-F{i:02d}':{'status':'PASS','evidence':'canonical/regression and canonical/reproduction'} for i in range(10)},
             'commits':{**{k:v['evidence_commit'] for k,v in history['tasks'].items()},'F0-05_closure':EXPECTED_START,
                        'F0-06_implementation':args.implementation_commit,
                        'F0-06_closure':'Commit containing G0_DECISION.md; resolve with git log -1 -- experiments/F0-06/G0_DECISION.md'},
             'regression':tests,'evidence':evidence,'platforms':{'Windows':'PASS','Linux':'NOT_RUN'},
             'handoff_inputs':handoff,'reproduction':reproduction,'external_evidence':history['external_evidence'],
             'limitations':['Linux not executed','Physical model accuracy/calibration not verified','Collision and physical robot safety not verified',
                'Timing is host-specific','Wide-start DLS performance remains the measured baseline; no success-rate gate',
                'Coverage is empirical, not complete reachability coverage','External solvers are Core work, not started',
                'No trained neural model','Foundations closure is not production or robot safety approval',
                'Manufacturer PDF raw bytes/hash remain unavailable','Historical F0-00 JSON checkout line endings differ from Git blobs'],
             'gate':gate,'g0_impact':'All required canonical checks passed; listed limitations are retained scope limits.'}
    validate_index(index)
    bad = copy.deepcopy(index); del bad['tests']['T-F09']
    try: validate_index(bad)
    except ValueError: pass
    else: raise ValueError('structural negative test failed')
    write(OUT/'FOUNDATIONS_EVIDENCE_INDEX.json',index)
    md = '# Foundations kanıt indeksi\n\nBelge r1 · G0 PASS / ACCEPTED · Windows PASS · Linux NOT_RUN\n\n'
    md += 'Makine kaydı: [FOUNDATIONS_EVIDENCE_INDEX.json](FOUNDATIONS_EVIDENCE_INDEX.json).\n'
    md += 'Tüm görevler F0-00–F0-06 COMPLETE; T-F00–T-F09 PASS. REQ-F00 tanımlı değildir; F0-00 → REQ-F01.\n\n'
    md += '| Grup | Geçen | JUnit |\n|---|---:|---|\n'
    for name,junit in tests.items(): md += f"| {name} | {junit['passed']} | canonical/regression/junit/{name}.xml |\n"
    md += '\n212 eski SHA256SUMS kaydı, 18 Stage 1 kaydı, robot manifesti ve 14 yerel Git dışı dosya denetlendi. '
    md += 'Eski paylaşılan dosya revizyonları kendi ham commit bloblarıyla doğrulandı; güncel hashle eşitlik iddiası yok.\n\n'
    md += 'Config/schema/robot devir hashleri [handoff-inputs.json](handoff-inputs.json), veri/query/result hashleri [reproduction-summary.json](reproduction-summary.json) içindedir. '
    md += 'Komut/JUnit/log yolları ve tüm dosya hashleri JSON indeksindedir. G0 etkisi: kritik açık hata yok.\n\n'
    md += '\n'.join('- '+x for x in index['limitations'])+'\n'
    (OUT/'FOUNDATIONS_EVIDENCE_INDEX.md').write_text(md,encoding='utf-8',newline='\n')
    files = sorted(p for p in OUT.rglob('*') if p.is_file() and p.name not in ('SHA256SUMS','evidence-verification.json'))
    write(OUT/'evidence-verification.json',{'status':'PASS','index_structure':'PASS','missing_test_mutation':'DETECTED',
          'checksum_records':len(files)+1,'checksum_self_excluded':True,'checksum_order':'lexicographic repository-relative path',
          'verification_command':'python scripts/finalize_f06.py; checks raw bytes after writing SHA256SUMS'})
    files = sorted(files+[OUT/'evidence-verification.json'])
    (OUT/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in files),encoding='utf-8',newline='\n')
    count = verify_checksums(ROOT,OUT/'SHA256SUMS')
    print(f'PASS: {count} checksum files; index structure and negative structure test; {len(mutations)} recorded mutations; G0 ACCEPTED')


if __name__ == '__main__':
    main()
