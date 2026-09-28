"""An old reader must never deliver lines into a replacement process queue."""
import io
import json
from types import SimpleNamespace

from neurokinematics.core import worker


def test_old_reader_retains_its_process_queue(monkeypatch, tmp_path):
    readers = []
    processes = []
    ready = json.dumps({'ready': True, 'solver_id': 'test', 'solver_config_sha256': 'hash'}) + '\n'

    class Thread:
        def __init__(self, target, daemon): self.target = target
        def start(self):
            readers.append(self.target)
            self.target()
        def join(self, timeout): pass

    def popen(*args, **kwargs):
        process = SimpleNamespace(stdout=io.StringIO(ready), stdin=io.StringIO(),
                                  poll=lambda: 0, wait=lambda timeout: 0)
        processes.append(process)
        return process

    monkeypatch.setattr(worker.threading, 'Thread', Thread)
    monkeypatch.setattr(worker.subprocess, 'Popen', popen)
    client = worker.LocalWorker(['fake'], 'test', 'hash', tmp_path / 'stderr.log')
    client.start()
    old_queue = client.lines
    # Simulate a delayed old reader while a replacement is already installed.
    client.process = None
    client._stderr_file.close()
    client.start()
    new_queue = client.lines
    previous_size = new_queue.qsize()
    processes[0].stdout.seek(0)
    readers[0]()
    assert new_queue.qsize() == previous_size
    assert old_queue.qsize() == 3
    client.stop()
    processes[0].stdout.close()
    processes[0].stdin.close()
