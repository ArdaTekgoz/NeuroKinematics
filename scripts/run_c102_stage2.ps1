param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('pilot', 'full', 'verify', 'environment')]
    [string]$Stage,
    [string]$Output = 'data/generated/C1-02/v1'
)

$ErrorActionPreference = 'Stop'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'

switch ($Stage) {
    'pilot' { pixi run --locked python scripts/run_c102_pilot.py }
    'full' { pixi run --locked python scripts/run_c102_full.py --output $Output }
    'verify' { pixi run --locked python scripts/verify_c102.py --input $Output }
    'environment' { pixi run --locked python scripts/record_c102_environment.py }
}
if ($LASTEXITCODE -ne 0) { throw "C1-02 $Stage failed with exit code $LASTEXITCODE" }
