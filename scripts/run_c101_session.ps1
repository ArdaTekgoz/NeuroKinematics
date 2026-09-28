[CmdletBinding()]
param(
    [ValidateSet('prepare', 'smoke', 'pilot', 'full', 'verify')]
    [string]$Stage = 'prepare',

    [ValidateLength(1, 48)]
    [ValidatePattern('(?-i)^[a-z0-9]+(?:-[a-z0-9]+)*$')]
    [string]$SessionName = 'udp-v1',

    [string]$EnvironmentLock = 'experiments\C1-01\runtime-build-v1\environment-lock.json'
)

$ErrorActionPreference = 'Stop'
$Stage = $Stage.ToLowerInvariant()
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$evidenceDirectory = Join-Path $taskRoot 'experiments\C1-01'
$scriptsDirectory = Join-Path $taskRoot 'scripts'
$sourceDirectory = Join-Path $taskRoot 'src'
$testsDirectory = Join-Path $taskRoot 'tests'
$sessionScript = Join-Path $scriptsDirectory 'c101_session.py'
$sessionDirectory = Join-Path $evidenceDirectory $SessionName
$snapshotDirectory = Join-Path $sessionDirectory 'inputs'
$snapshotSource = Join-Path $snapshotDirectory 'src'
$snapshotTests = Join-Path $snapshotDirectory 'tests'
$snapshotScripts = Join-Path $snapshotDirectory 'scripts'
$lockPath = Join-Path $sessionDirectory 'base-environment-lock.json'
if ($Stage -eq 'prepare') {
    $lockPath = if ([IO.Path]::IsPathRooted($EnvironmentLock)) { $EnvironmentLock } else { Join-Path $taskRoot $EnvironmentLock }
}
if (-not (Test-Path -LiteralPath $lockPath -PathType Leaf)) {
    throw "Required environment lock missing: $lockPath"
}
if ($Stage -eq 'prepare') {
    if (Test-Path -LiteralPath $sessionDirectory) {
        throw "Session already exists; prepare requires a new SessionName: $sessionDirectory"
    }
    foreach ($requiredDirectory in @($sourceDirectory, $testsDirectory)) {
        if (-not (Test-Path -LiteralPath $requiredDirectory -PathType Container)) {
            throw "Snapshot source directory missing: $requiredDirectory"
        }
    }
    if (-not (Test-Path -LiteralPath $sessionScript -PathType Leaf)) {
        throw "Required session script missing: $sessionScript"
    }
} else {
    foreach ($requiredDirectory in @($snapshotSource, $snapshotTests, $snapshotScripts)) {
        if (-not (Test-Path -LiteralPath $requiredDirectory -PathType Container)) {
            throw "Prepared session input missing: $requiredDirectory"
        }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $snapshotScripts 'c101_session.py') -PathType Leaf)) {
        throw 'Prepared session script missing; run prepare with a new SessionName.'
    }
}

$dockerExecutable = (Get-Command docker.exe -CommandType Application -ErrorAction Stop | Select-Object -First 1).Path
$environmentRecord = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json
$imageId = [string]$environmentRecord.image_id
if ($imageId -cnotmatch '^sha256:[0-9a-f]{64}$') {
    throw "environment-lock.json must contain an immutable Docker image ID: $imageId"
}

# PowerShell 5 wraps native stderr in ErrorRecord objects. Check Docker's exit
# code explicitly so ordinary progress on stderr does not abort the command.
$previousPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    $imageInspection = & $dockerExecutable image inspect $imageId
    $inspectionExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previousPreference
}
if ($inspectionExitCode -ne 0) {
    throw "Docker image inspection failed (exit $inspectionExitCode): $imageId"
}
$inspectedImages = @($imageInspection | ConvertFrom-Json)
if ($inspectedImages.Count -ne 1) {
    throw 'Expected exactly one image from Docker image inspection.'
}
$inspectedImage = $inspectedImages[0]
if ($inspectedImage.Id -cne $imageId) {
    throw "Docker image ID differs from environment-lock.json: $($inspectedImage.Id)"
}
if ($inspectedImage.Os -cne 'linux' -or $inspectedImage.Architecture -cne 'amd64') {
    throw "Linux amd64 image required; found: $($inspectedImage.Os)/$($inspectedImage.Architecture)"
}

$attemptId = '{0}-{1}' -f (Get-Date -Format 'yyyyMMdd-HHmmss-fff'), ([guid]::NewGuid().ToString('N'))
$inspectionPath = Join-Path $evidenceDirectory "$SessionName-$Stage-docker-inspect-$attemptId.json"
$logPath = Join-Path $evidenceDirectory "$SessionName-$Stage-$attemptId.log"
foreach ($newPath in @($inspectionPath, $logPath)) {
    if (Test-Path -LiteralPath $newPath) {
        throw "Refusing to overwrite existing attempt evidence: $newPath"
    }
}
$imageInspection | Set-Content -LiteralPath $inspectionPath -Encoding UTF8

function Copy-PythonSnapshot {
    param([string]$SourceDirectory, [string]$DestinationDirectory)
    # Reject links so the snapshot is made only from files inside this repo tree.
    $sourceRoot = (Get-Item -LiteralPath $SourceDirectory -Force).FullName.TrimEnd('\')
    $sourceItems = @(Get-Item -LiteralPath $sourceRoot -Force) + @(Get-ChildItem -LiteralPath $sourceRoot -Recurse -Force)
    foreach ($sourceItem in $sourceItems) {
        if (($sourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Linked snapshot input is unsupported: $($sourceItem.FullName)"
        }
    }
    New-Item -ItemType Directory -Path $DestinationDirectory -ErrorAction Stop | Out-Null
    foreach ($sourceFile in $sourceItems) {
        if ($sourceFile.PSIsContainer -or $sourceFile.Extension -cne '.py') { continue }
        $relativePath = $sourceFile.FullName.Substring($sourceRoot.Length + 1)
        $destinationFile = Join-Path $DestinationDirectory $relativePath
        $destinationParent = Split-Path -Parent $destinationFile
        if (-not (Test-Path -LiteralPath $destinationParent)) {
            New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
        }
        Copy-Item -LiteralPath $sourceFile.FullName -Destination $destinationFile -ErrorAction Stop
    }
}

if ($Stage -eq 'prepare') {
    New-Item -ItemType Directory -Path $sessionDirectory -ErrorAction Stop | Out-Null
    Copy-Item -LiteralPath $lockPath -Destination (Join-Path $sessionDirectory 'base-environment-lock.json')
    New-Item -ItemType Directory -Path $snapshotDirectory -ErrorAction Stop | Out-Null
    Copy-PythonSnapshot -SourceDirectory $sourceDirectory -DestinationDirectory $snapshotSource
    Copy-PythonSnapshot -SourceDirectory $testsDirectory -DestinationDirectory $snapshotTests
    New-Item -ItemType Directory -Path $snapshotScripts -ErrorAction Stop | Out-Null
    Copy-Item -LiteralPath $sessionScript -Destination (Join-Path $snapshotScripts 'c101_session.py')
    Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $snapshotScripts 'run_c101_session.ps1')
}

function Invoke-DockerLogged {
    param([string[]]$DockerArguments, [string]$LogPath)
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $dockerExecutable @DockerArguments 2>&1 | Tee-Object -FilePath $LogPath -ErrorAction Stop
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) {
        throw "C1-01 $Stage failed (Docker exit $exitCode); see $LogPath"
    }
}

$dockerArguments = @(
    'run', '--rm', '--name', 'c101-main-session',
    '--cpuset-cpus', '0,1',
    '--memory', '8g', '--memory-swap', '8g',
    '-e', 'RMW_IMPLEMENTATION=rmw_fastrtps_cpp',
    '-e', 'FASTDDS_BUILTIN_TRANSPORTS=UDPv4',
    '-e', 'OMP_NUM_THREADS=1',
    '-e', 'OPENBLAS_NUM_THREADS=1',
    '-e', 'MKL_NUM_THREADS=1',
    '-e', 'NUMEXPR_NUM_THREADS=1',
    '-e', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1',
    '-e', 'PYTHONDONTWRITEBYTECODE=1',
    '-e', 'PIXI_NO_INSTALL=true',
    '-e', 'C101_SESSION_SCRIPT=/diagnostics/c101_session.py',
    '--mount', "type=bind,source=$evidenceDirectory,target=/evidence",
    '--mount', "type=bind,source=$snapshotSource,target=/work/src,readonly",
    '--mount', "type=bind,source=$snapshotScripts,target=/diagnostics,readonly",
    '--mount', "type=bind,source=$snapshotTests,target=/work/tests,readonly",
    $imageId, 'python', '/diagnostics/c101_session.py', $Stage,
    '--session', "/evidence/$SessionName", '--image-id', $imageId
)
Write-Host "C1-01 stage: $Stage; session: $SessionName; image: $imageId"
Write-Host "Attempt log: $logPath"
Invoke-DockerLogged -DockerArguments $dockerArguments -LogPath $logPath
Write-Host "C1-01 $Stage completed. Evidence directory: $evidenceDirectory\$SessionName"
