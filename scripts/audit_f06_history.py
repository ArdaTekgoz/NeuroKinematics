"""Raw-byte audit of historical Foundations evidence, retaining supersession details."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from run_f06_regression import write


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = Path.cwd()
    out = root/'experiments/F0-06'
    records, errors, tasks = [], [], {}
    # These shared files evolved in later accepted tasks. Historical blob hashes
    # are checked explicitly; immutable robot/config/schema files get no exception.
    mutable = {'pixi.toml', 'docs/SETUP.md', '.gitattributes'}
    overlay = json.loads((out/'overlay.json').read_bytes())
    for n in range(6):
        task = f'F0-0{n}'
        sums = root/'experiments'/task/'SHA256SUMS'
        commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', str(sums)], text=True).strip()
        tasks[task] = {'evidence_commit': commit, 'run_report': f'experiments/{task}/RUN_REPORT.md'}
        seen = set()
        for line in sums.read_text(encoding='utf-8').splitlines():
            expected, relative = line.split(maxsplit=1)
            path = root/relative
            rec = {'task': task, 'checksum_file': sums.relative_to(root).as_posix(), 'path': relative,
                   'expected_sha256': expected, 'historical_commit': commit}
            if relative in seen or not path.resolve().is_relative_to(root) or not path.is_file():
                rec['status'] = 'FAIL_MISSING_DUPLICATE_OR_PATH'
            else:
                actual = digest(path)
                rec['current_sha256'] = actual
                if actual == expected:
                    rec['status'] = 'PASS_CURRENT_RAW_BYTES'
                elif relative in overlay['files'] and actual == overlay['files'][relative]:
                    raw = subprocess.check_output(['git', 'show', f"{overlay['base_commit']}:{relative}"])
                    rec['baseline_blob_sha256'] = hashlib.sha256(raw).hexdigest()
                    rec['status'] = 'PASS_BASELINE_RAW_BLOB_F006_OVERLAY' if rec['baseline_blob_sha256'] == expected else 'FAIL_HASH'
                    rec['current_revision_note'] = 'Explicit F0-06 reproduction adapter, separately hashed in overlay.json'
                elif relative in mutable or (task == 'F0-00' and relative == 'pixi.lock'):
                    raw = subprocess.check_output(['git', 'show', f'{commit}:{relative}'])
                    rec['historical_blob_sha256'] = hashlib.sha256(raw).hexdigest()
                    rec['status'] = 'PASS_HISTORICAL_RAW_BLOB' if rec['historical_blob_sha256'] == expected else 'FAIL_HASH'
                    rec['later_changes'] = subprocess.check_output(['git', 'log', '--format=%H %s', f'{commit}..HEAD', '--', relative], text=True).splitlines()
                    if not rec['later_changes']:
                        rec['status'] = 'FAIL_UNEXPLAINED_CHANGE'
                else:
                    rec['status'] = 'FAIL_HASH'
            seen.add(relative)
            records.append(rec)
            if not rec['status'].startswith('PASS'):
                errors.append(rec)
    external = []
    dataset = json.loads((root/'experiments/F0-04/dataset-manifest.json').read_bytes())
    for subset, shards in dataset['shards'].items():
        for shard in shards:
            p = root/dataset['generation_path']/shard['path']
            external.append({'path': p.relative_to(root).as_posix(), 'expected_sha256': shard['file_sha256'],
                             'kind': 'F0-04 '+subset})
    for name in ('query', 'result'):
        manifest = json.loads((root/f'experiments/F0-05/{name}-manifest.json').read_bytes())
        external.append({'path': manifest['file'], 'expected_sha256': manifest['file_sha256'], 'kind': 'F0-05 '+name})
    for entry in external:
        path = root/entry['path']
        entry['tracked'] = subprocess.run(['git', 'ls-files', '--error-unmatch', entry['path']], capture_output=True).returncode == 0
        entry['actual_sha256'] = digest(path) if path.is_file() else None
        entry['status'] = ('PASS_LOCAL_IGNORED' if entry['actual_sha256'] == entry['expected_sha256']
                           else 'MISSING_NOT_VERIFIED' if not path.exists() else 'FAIL_HASH')
        if entry['status'] == 'FAIL_HASH': errors.append(entry)
    model = json.loads((root/'assets/robots/robot_a/manifest.json').read_bytes())
    model_files = []
    for item in model['files']:
        p = root/item['path']
        entry = {'path': item['path'], 'expected_sha256': item['sha256'], 'actual_sha256': digest(p) if p.is_file() else None}
        entry['status'] = 'PASS' if entry['expected_sha256'] == entry['actual_sha256'] else 'FAIL'
        model_files.append(entry)
        if entry['status'] != 'PASS': errors.append(entry)
    junit = []
    for p in sorted((root/'experiments').glob('F0-0[0-5]/*junit.xml')):
        suites = ET.parse(p).getroot().findall('testsuite')
        junit.append({'path': p.relative_to(root).as_posix(), 'sha256': digest(p),
                      **{k: sum(int(s.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}})
    expected_counts = {'F0-00/pytest':6,'F0-01/pytest':16,'F0-02/pytest':102,'F0-03/pytest':159,
                       'F0-03/metrics':48,'F0-04/f04-unit':10,'F0-04/tf05':3,'F0-04/tf06':2,
                       'F0-04/tf07':7,'F0-04/mutation':17,'F0-05/f05-unit':127,'F0-05/tf08':16,'F0-05/mutation':32}
    comparison = []
    for key, count in expected_counts.items():
        item = next(x for x in junit if x['path'] == f'experiments/{key}-junit.xml')
        ok = item['tests'] == count and not any(item[k] for k in ('failures','errors','skipped'))
        comparison.append({'path':item['path'],'report_count':count,'junit_count':item['tests'],'status':'PASS' if ok else 'FAIL'})
        if not ok: errors.append(comparison[-1])
    result = {'status':'PASS' if not errors else 'FAIL','audited_utc':datetime.now(timezone.utc).isoformat(),
              'tasks':tasks,'checksums':records,'external_evidence':external,'robot_manifest':model_files,
              'junit_inventory':junit,'report_count_comparisons':comparison,'errors':errors,
              'notes':['Historical shared-file revisions are checked as raw Git blobs; they are not asserted equal to current files.',
                       'F0-00 raw environment/source JSON matches local checksum bytes. Git originally normalized their line endings; fresh checkout verification is separately required.',
                       'Development failure JUnit records are retained and are not treated as final acceptance.']}
    write(out/'history-audit.json', result)
    print(result['status'], len(records), 'checksum records;',len(external),'external files;',len(errors),'errors')
    if errors: raise SystemExit(1)


if __name__ == '__main__':
    main()
