"""Run legacy test directories in separate processes (shared module basenames)."""
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/C1-06R/diagnostic5'


def main():
    total=0
    for group in ('f0_01','f0_02','f0_03','c1_03','c1_04','c1_05','c1_06r'):
        output=BASE/(group+'-tests.xml')
        if output.exists():raise FileExistsError(output)
        argv=[sys.executable,'-m','pytest','tests/'+group,'-q','--junitxml='+str(output)]
        print('COMMAND:',repr(argv),flush=True)
        subprocess.run(argv,cwd=ROOT,check=True)
        suites=ET.parse(output).getroot().findall('testsuite')
        assert all(int(x.attrib.get(k,0))==0 for x in suites for k in ('failures','errors','skipped'))
        total+=sum(int(x.attrib['tests']) for x in suites)
    print(f'PASS:{total} tests across seven isolated processes',flush=True)


if __name__=='__main__':main()
