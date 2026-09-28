"""Read-only comparison of a prepared C1-01 runtime; no measured solver calls."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


def differences(expected, actual, path='$'):
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = []
        for key in sorted(expected.keys() | actual.keys()):
            child = path + '.' + key
            if key not in expected or key not in actual:
                result.append({'field': child, 'prepared': expected.get(key),
                               'current': actual.get(key), 'missing_key': True})
            else:
                result.extend(differences(expected[key], actual[key], child))
        return result
    if expected != actual:
        return [{'field': path, 'prepared': expected, 'current': actual}]
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--smoke-startup', action='store_true',
                        help='Use the prepared main startup and stop immediately after runtime capture.')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('prepared_c101_session', '/diagnostics/c101_session.py')
    prepared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepared)
    locked = prepared.read_json(args.session / 'runtime-lock.json')
    if args.smoke_startup:
        current = capture_smoke_startup(prepared, args.session, args.image_id)
    else:
        current = prepared.capture_runtime(args.session, args.image_id)
    changed = differences(locked, current)
    print(json.dumps({'scope': 'read-only runtime diagnostic; no smoke or benchmark',
                      'status': 'DRIFT' if changed else 'MATCH',
                      'startup': 'prepared smoke main; stopped before measurements' if args.smoke_startup else 'capture only',
                      'runtime_lock_sha256': prepared.sha256(args.session / 'runtime-lock.json'),
                      'differences': changed}, indent=2), flush=True)
    return 0


def capture_smoke_startup(prepared, session, image_id):
    """Run the original startup, stopping before comparison or solver dispatch."""
    class CaptureFinished(BaseException):
        pass

    namespace = prepared.main.__globals__
    original = namespace['capture_runtime']
    captured = []

    def capture_and_stop(*args, **kwargs):
        captured.append(original(*args, **kwargs))
        raise CaptureFinished()

    old_argv = sys.argv
    namespace['capture_runtime'] = capture_and_stop
    sys.argv = ['/diagnostics/c101_session.py', 'smoke', '--session', str(session), '--image-id', image_id]
    try:
        try:
            prepared.main()
        except CaptureFinished:
            return captured[0]
        raise RuntimeError('Prepared startup unexpectedly returned without runtime capture')
    finally:
        namespace['capture_runtime'] = original
        sys.argv = old_argv


if __name__ == '__main__':
    raise SystemExit(main())
