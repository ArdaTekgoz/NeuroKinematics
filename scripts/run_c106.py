"""Run authorized C1-06 gates in order; no training or model selection."""
import argparse
from pathlib import Path
from neurokinematics.neural import c106_runtime as runtime
from neurokinematics.neural import c106_reporting as report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['preflight','identity','evaluate','summarize','audit'])
    p.add_argument('--config',type=Path,default=runtime.BASE/'config.json')
    p.add_argument('--approval',type=Path,default=runtime.STAGE2/'approval.json')
    p.add_argument('--run',default='final-001')
    a=p.parse_args()
    if a.config.resolve()!=(runtime.BASE/'config.json').resolve(): p.error('only frozen config allowed')
    if a.approval.resolve()!=(runtime.STAGE2/'approval.json').resolve(): p.error('only recorded user authorization allowed')
    runtime.setup()
    if a.action=='preflight': runtime.preflight()
    elif a.action=='identity': runtime.identity(a.run)
    elif a.action=='evaluate': runtime.evaluate(a.run)
    elif a.action=='summarize': report.summarize(a.run)
    else: report.audit(a.run)


if __name__=='__main__': main()
