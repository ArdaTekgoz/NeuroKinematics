"""One real F0-06 public-API reproduction, using separate frozen smoke inputs."""
import argparse
import json
from pathlib import Path

from neurokinematics.data.factory import generate_dataset
from neurokinematics.benchmark.queries import generate_queries, verify_queries
from neurokinematics.benchmark.runner import benchmark, verify_results, write_json
from neurokinematics.foundations_gate import audit_dataset, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config-root', type=Path, default=Path('experiments/F0-06'))
    args = parser.parse_args()
    root = args.output
    if root.exists():
        raise ValueError('choose a fresh output directory')
    cfg = json.loads((args.config_root/'config.json').read_bytes())
    for path, expected in cfg['config_sha256'].items():
        if sha(args.config_root/path) != expected:
            raise ValueError('smoke config changed after freeze')
    dataset = root/'dataset'
    evidence = root/'data-evidence'
    data_config = args.config_root/'data-config.json'
    manifest = generate_dataset(dataset, evidence, config_path=data_config)
    # The factory's legacy manifest command assumes default production config.
    # Keep its raw output; the exact smoke command is recorded by this entrypoint.
    write_json(root/'dataset-audit.json', audit_dataset(dataset, manifest, data_config, evidence))
    options = {'config_path': args.config_root/'benchmark-config.json',
               'dataset_manifest': dataset/'dataset-manifest.json', 'data_config': data_config}
    query = root/'query-list.jsonl'
    query_manifest = generate_queries(query, dataset, **options)
    write_json(root/'query-manifest.json', query_manifest)
    write_json(root/'query-verification.json', verify_queries(query, dataset, query_manifest, **options))
    measured = benchmark(query, root/'results.jsonl', root/'sample.jsonl', query_manifest, dataset, **options)
    write_json(root/'measurement-summary.json', measured)
    verified = verify_results(query, root/'results.jsonl', dataset, query_manifest,
                              expected_file_hash=measured['result_file_sha256'], **options)
    if measured['groups'] != verified['groups']:
        raise ValueError('measured and independently aggregated groups differ')
    write_json(root/'result-verification.json', verified)
    write_json(root/'summary.json', {'status': 'PASS', 'dataset_content_sha256': manifest['dataset_content_sha256'],
               'query_sha256': query_manifest['query_list_sha256'], 'dataset_rows': sum(manifest['record_counts'].values()),
               'query_rows': query_manifest['record_count'], 'result_rows': verified['record_count'],
               'result_sha256': verified['result_file_sha256'], 'config_sha256': cfg['config_sha256']})
    print(json.dumps(json.loads((root/'summary.json').read_bytes())), flush=True)


if __name__ == '__main__':
    main()
