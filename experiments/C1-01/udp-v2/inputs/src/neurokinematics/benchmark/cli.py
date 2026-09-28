"""F0-05 query, result and summary CLI. Nonzero on the first failed check."""

import argparse
import json
from pathlib import Path
import sys

from .queries import generate_queries,verify_queries
from .runner import benchmark,verify_results,write_json


def _manifest(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('generate');g.add_argument('--output',type=Path,required=True)
    g.add_argument('--dataset-root',type=Path,required=True)
    g.add_argument('--manifest-out',type=Path)
    v=sub.add_parser('verify-queries');v.add_argument('--input',type=Path,required=True)
    v.add_argument('--dataset-root',type=Path,required=True);v.add_argument('--manifest',type=Path,required=True)
    b=sub.add_parser('benchmark');b.add_argument('--queries',type=Path,required=True)
    b.add_argument('--dataset-root',type=Path,required=True);b.add_argument('--manifest',type=Path,required=True)
    b.add_argument('--results',type=Path,required=True);b.add_argument('--sample',type=Path,required=True)
    b.add_argument('--summary-out',type=Path,required=True)
    r=sub.add_parser('verify-results');r.add_argument('--queries',type=Path,required=True)
    r.add_argument('--results',type=Path,required=True);r.add_argument('--dataset-root',type=Path,required=True)
    r.add_argument('--manifest',type=Path,required=True);r.add_argument('--expected-sha256')
    r.add_argument('--summary-out',type=Path)
    s=sub.add_parser('summarize');s.add_argument('--verified-summary',type=Path,required=True)
    s.add_argument('--measurement-summary',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    try:
        if args.command=='generate':
            result=generate_queries(args.output,args.dataset_root)
            if args.manifest_out:write_json(args.manifest_out,result)
        elif args.command=='verify-queries':
            result=verify_queries(args.input,args.dataset_root,_manifest(args.manifest))
        elif args.command=='benchmark':
            if args.results.exists() or args.sample.exists():
                raise ValueError('benchmark output already exists; choose a new output path')
            result=benchmark(args.queries,args.results,args.sample,_manifest(args.manifest),args.dataset_root,
                             progress=lambda deadline,pass_index,count:print(f'{deadline}ms pass {pass_index}: {count} records',flush=True))
            write_json(args.summary_out,result)
        elif args.command=='verify-results':
            result=verify_results(args.queries,args.results,args.dataset_root,_manifest(args.manifest),
                                  expected_file_hash=args.expected_sha256)
            if args.summary_out:write_json(args.summary_out,result)
        else:
            verified=_manifest(args.verified_summary);measured=_manifest(args.measurement_summary)
            if verified['status']!='PASS' or verified['result_file_sha256']!=measured['result_file_sha256']:
                raise ValueError('measurement/result verification mismatch')
            if verified['groups']!=measured['groups']:
                raise ValueError('measured and independently re-aggregated summaries differ')
            result={'status':'PASS','query_list_sha256':measured['query_list_sha256'],
                    'result_file_sha256':verified['result_file_sha256'],
                    'record_count':verified['record_count'],'groups':verified['groups'],
                    'warmup_calls':measured['warmup_calls'],'warmup_elapsed_ns':measured['warmup_elapsed_ns'],
                    'query_loading_ns':measured['query_loading_ns'],
                    'serialization_io_ns':measured['serialization_io_ns']}
            write_json(args.output,result)
        print(json.dumps(result if args.command!='benchmark' else
                         {k:v for k,v in result.items() if k!='groups'},ensure_ascii=False,allow_nan=False))
        return 0
    except (ValueError,RuntimeError,OSError,KeyError,TypeError) as exc:
        print(f'F0-05 {args.command} FAIL: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__':raise SystemExit(main())
