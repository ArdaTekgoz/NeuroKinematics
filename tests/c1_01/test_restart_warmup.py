"""A replacement process must finish warmup before its next measured call."""
from neurokinematics.core import runner
from neurokinematics.core.contract import load_contract


def test_restart_warms_before_next_measurement(monkeypatch, tmp_path):
    config = load_contract()
    rows, manifest = runner.load_queries(runner.FROZEN_QUERY_PATH, config)
    config['benchmark']['measurement_passes'] = 1
    config['benchmark']['deadline_profiles_ms'] = [10]
    events = []

    class Worker:
        process = None
        def __init__(self, *args): pass
        def start(self):
            self.process = object()
            events.append('start')
            return 1
        def call(self, request, *, late_reply_window_s):
            assert late_reply_window_s == 3.0
            events.append('warm')
        def stop(self): self.process = None

    def measured(*args):
        events.append('measure')
        if events.count('measure') == 1:
            args[5].process = None
        return {'common_status': 'TIMEOUT', 'profile_a_geometry': False,
                'profile_b_geometry': False, 'profile_a_deadline': False, 'profile_b_deadline': False}

    monkeypatch.setattr(runner, 'LocalWorker', Worker)
    monkeypatch.setattr(runner, 'evaluate_attempt', measured)
    monkeypatch.setattr(runner, 'validate_result_record', lambda *args: None)
    result = runner.run_method(config['solvers'][0], config, rows[:2], manifest,
                               tmp_path, None, rows[:20], mode='benchmark')
    assert events == ['start'] + ['warm'] * 20 + ['measure', 'start'] + ['warm'] * 20 + ['measure']
    assert result['warmup_calls'] == 40
    assert result['restart_warmups'] == 1
