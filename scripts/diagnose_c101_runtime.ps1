[CmdletBinding()]
param(
    [ValidatePattern('(?-i)^[a-z0-9]+(?:-[a-z0-9]+)*$')]
    [string]$SessionName = 'udp-v1'
)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$evidence = Join-Path $taskRoot 'experiments\C1-01'
$session = Join-Path $evidence $SessionName
$inputs = Join-Path $session 'inputs'
$imageId = [string](Get-Content -LiteralPath (Join-Path $session 'base-environment-lock.json') -Raw | ConvertFrom-Json).image_id
if ($imageId -cnotmatch '^sha256:[0-9a-f]{64}$') { throw 'Invalid prepared image identity.' }
$dockerExecutable = (Get-Command docker.exe -CommandType Application -ErrorAction Stop | Select-Object -First 1).Path
$logPath = Join-Path $evidence ("$SessionName-runtime-diagnostic-" + [guid]::NewGuid().ToString('N') + '.log')
$dockerArguments = @(
    'run', '--rm', '--name', 'c101-main-session', '--cpuset-cpus', '0,1',
    '-e', 'RMW_IMPLEMENTATION=rmw_fastrtps_cpp',
    '-e', 'FASTDDS_BUILTIN_TRANSPORTS=UDPv4',
    '-e', 'OMP_NUM_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1',
    '-e', 'MKL_NUM_THREADS=1', '-e', 'NUMEXPR_NUM_THREADS=1',
    '-e', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1', '-e', 'PYTHONDONTWRITEBYTECODE=1',
    '-e', 'PIXI_NO_INSTALL=true',
    '-e', 'C101_SESSION_SCRIPT=/diagnostics/c101_session.py',
    '--mount', "type=bind,source=$evidence,target=/evidence",
    '--mount', "type=bind,source=$inputs\src,target=/work/src,readonly",
    '--mount', "type=bind,source=$inputs\tests,target=/work/tests,readonly",
    '--mount', "type=bind,source=$inputs\scripts,target=/diagnostics,readonly",
    '--mount', "type=bind,source=$PSScriptRoot\diagnose_c101_runtime.py,target=/runtime-diagnostic.py,readonly",
    $imageId, 'python', '/runtime-diagnostic.py',
    '--session', "/evidence/$SessionName", '--image-id', $imageId, '--smoke-startup'
)
$runtimeRecord = Get-Content -LiteralPath (Join-Path $session 'runtime-lock.json') -Raw | ConvertFrom-Json
if ($null -ne $runtimeRecord.memory_policy) {
    $dockerArguments = @($dockerArguments[0]) + @('--memory', '8g', '--memory-swap', '8g') + @($dockerArguments[1..($dockerArguments.Length - 1)])
}
Write-Host "Smoke startup diagnostic; no solver measurements. Log: $logPath"
$previousPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    & $dockerExecutable @dockerArguments 2>&1 | Tee-Object -FilePath $logPath -ErrorAction Stop
    $exitCode = $LASTEXITCODE
} finally { $ErrorActionPreference = $previousPreference }
if ($exitCode -ne 0) { throw "Runtime diagnostic failed (exit $exitCode); see $logPath" }
