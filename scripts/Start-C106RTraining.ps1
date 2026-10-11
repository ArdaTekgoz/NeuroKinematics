param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$TaskRoot = Split-Path $PSScriptRoot -Parent
Push-Location -LiteralPath $TaskRoot
try {
    if (!(Test-Path -LiteralPath '.venv/c106r/Scripts/python.exe')) {
        throw 'C1-06R ortamı yok. experiments/C1-06R/USER_TRAINING.md kurulum bölümünü kullanın.'
    }
    & pixi run --locked .venv/c106r/Scripts/python.exe scripts/c106r_command.py user-preflight-$(Get-Date -Format 'yyyyMMdd-HHmmss') -- pixi run --locked .venv/c106r/Scripts/python.exe -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Paket denetimi başarısız.' }
    $env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
    $env:OMP_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:NUMEXPR_NUM_THREADS = '1'
    $env:PYTHONUTF8 = '1'
    & pixi run --locked .venv/c106r/Scripts/python.exe scripts/check_c106r_training.py
    if ($LASTEXITCODE -ne 0) { throw 'Eğitim paketi bütünlük kontrolü başarısız.' }
    if ($CheckOnly) { return }
    & pixi run --locked .venv/c106r/Scripts/python.exe scripts/run_c106r_round1.py --resume
    if ($LASTEXITCODE -ne 0) { throw 'Eğitim durdu. Mevcut checkpoint ve kayıtlar korundu; hata metnini paylaşın.' }
} finally {
    Pop-Location
}
