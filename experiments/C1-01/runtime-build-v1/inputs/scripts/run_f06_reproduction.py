"""Capture two independent F0-06 API/CLI runs and their deterministic projection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from run_f06_regression import now, sha, write
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from neurokinematics.foundations_gate import require_same


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cwd', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.cwd = args.cwd.resolve()
    output = args.output.resolve()
    records = []
    for label in ('a', 'b'):
        relative = f'experiments/F0-06/reproduction/{label}'
        argv = ['pixi', 'run', '--locked', 'python', 'scripts/reproduce_f06.py', '--output', relative]
        entry = {'name': f'reproduce-{label}', 'argv':argv, 'command':subprocess.list2cmdline(argv),
                 'cwd':str(args.cwd), 'started_utc':now()}
        result = subprocess.run(argv, cwd=args.cwd, env=dict(os.environ,PYTHONIOENCODING='utf-8'), capture_output=True)
        entry.update(finished_utc=now(), exit_code=result.returncode)
        (output/'logs').mkdir(parents=True,exist_ok=True)
        for kind, content in [('stdout',result.stdout),('stderr',result.stderr)]:
            path = output/'logs'/f'reproduce-{label}.{kind}.log'
            path.write_bytes(content)
            entry[kind] = {'path':str(path),'sha256':sha(path)}
        records.append(entry)
        write(output/'reproduction-commands.json',records)
        print(label,result.returncode,result.stdout.decode('utf-8',errors='replace'),flush=True)
        if result.returncode:
            print(result.stderr.decode('utf-8',errors='replace'),flush=True)
            raise SystemExit(result.returncode)
        shutil.copytree(args.cwd/relative, output/'reproduction'/label)
    summaries = [json.loads((output/'reproduction'/label/'summary.json').read_bytes()) for label in ('a','b')]
    manifests = [json.loads((output/'reproduction'/label/'dataset/dataset-manifest.json').read_bytes()) for label in ('a','b')]
    for field in ('dataset_content_sha256','query_sha256','dataset_rows','query_rows','result_rows','config_sha256'):
        require_same(summaries[0][field], summaries[1][field], field)
    require_same(manifests[0]['shards'], manifests[1]['shards'], 'shard bytes/content/order')
    fields = ('query_id','query_group_id','solver_name','solver_version','solver_config_sha256',
              'query_list_sha256','deadline_profile_ms','measurement_pass_index','seeds','frame','tcp',
              'joint_order','quaternion_order','subset','start_class','target_source','q_current',
              'target_position_m','target_quaternion_wxyz','collision')
    projections = []
    for label in ('a','b'):
        rows = [json.loads(line) for line in (output/'reproduction'/label/'results.jsonl').read_bytes().splitlines()]
        projection = [{k: row[k] for k in fields} for row in rows]
        projections.append(hashlib.sha256(json.dumps(projection,sort_keys=True,separators=(',',':')).encode()).hexdigest())
        if any(row['joint_limits'] != 'PASS' for row in rows):
            raise ValueError('benchmark limit validation failed')
    require_same(projections[0], projections[1], 'benchmark deterministic identity')
    write(output/'reproduction-summary.json',{'status':'PASS','runs':summaries,
          'dataset_determinism':'PASS','query_determinism':'PASS','shard_determinism':'PASS',
          'deterministic_benchmark_projection_sha256':projections[0], 'projection_fields':fields,
          'result_schemas':'PASS','split_membership':'IDENTICAL_SHARD_BYTES',
          'manifest_binding':'Each result binds its own raw manifest; generation paths differ, so manifest raw hashes differ.',
          'timing':'Elapsed time, iterations and timeout-dependent outcomes vary with Windows scheduling and hardware; no raw result equality required.'})


if __name__ == '__main__':
    main()
