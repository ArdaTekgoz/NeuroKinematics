param([string]$ImageName = 'neurokinematics-c101:stage2')
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $taskRoot
$evidenceDirectory = Join-Path $taskRoot 'experiments\C1-01'
if (Test-Path -LiteralPath (Join-Path $evidenceDirectory 'environment-lock.json')) {
    $previousEvidence = Join-Path $evidenceDirectory ('attempts\' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
    New-Item -ItemType Directory -Path $previousEvidence -Force | Out-Null
    foreach ($name in @('environment-lock.json', 'docker-image-inspect.json', 'docker-build-evidence', 'linux-smoke', 'linux-pilot', 'linux-full', 'linux-portable-critical-regression.xml')) {
        $source = Join-Path $evidenceDirectory $name
        if (Test-Path -LiteralPath $source) { Copy-Item -LiteralPath $source -Destination $previousEvidence -Recurse }
    }
}
function Invoke-DockerLogged {
    param([string[]]$DockerArguments, [string]$LogPath)
    if (Test-Path -LiteralPath $LogPath) {
        $archivePath = '{0}.{1}.{2}.previous' -f $LogPath, (Get-Date -Format 'yyyyMMdd-HHmmss-fff'), ([guid]::NewGuid().ToString('N').Substring(0, 8))
        Copy-Item -LiteralPath $LogPath -Destination $archivePath
    }
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & docker @DockerArguments 2>&1 | Tee-Object -FilePath $LogPath
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) { throw "Docker command failed (exit $exitCode); see $LogPath" }
}
$queryPath = Join-Path $taskRoot 'data\generated\F0-05\acceptance\run-a\query-list.jsonl'
$queryHash = (Get-FileHash -LiteralPath $queryPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($queryHash -ne '120b41f07109aeaca10e10fbb04783167cdc4ba282e7976941468c7dccfa4976') {
    throw "Frozen query hash mismatch: $queryHash"
}
$serverPlatform = docker info --format '{{.OSType}}/{{.Architecture}}'
if ($LASTEXITCODE -ne 0 -or $serverPlatform -notin @('linux/x86_64', 'linux/amd64')) {
    throw "Linux x86_64 Docker server required; found: $serverPlatform"
}
Invoke-DockerLogged -DockerArguments @('pull', '--platform', 'linux/amd64', 'ros:jazzy-ros-base-noble') `
    -LogPath (Join-Path $evidenceDirectory 'docker-base-pull.log')
$baseInspection = docker image inspect ros:jazzy-ros-base-noble
if ($LASTEXITCODE -ne 0) { throw 'ROS base inspection failed' }
$baseInspection | Set-Content -LiteralPath (Join-Path $evidenceDirectory 'docker-base-inspect.json') -Encoding UTF8
$baseImage = ($baseInspection | ConvertFrom-Json)[0].RepoDigests[0]
if ($baseImage -notmatch '@sha256:[0-9a-f]{64}$') { throw "Immutable base digest missing: $baseImage" }
Invoke-DockerLogged -DockerArguments @('build', '--platform', 'linux/amd64', '--progress', 'plain',
    '--build-arg', "ROS_BASE_IMAGE=$baseImage", '-f', 'experiments/C1-01/Dockerfile', '-t', $ImageName, '.') `
    -LogPath (Join-Path $evidenceDirectory 'docker-build.log')
$imageInspection = docker image inspect $ImageName
if ($LASTEXITCODE -ne 0) { throw 'Built image inspection failed' }
$imageInspection | Set-Content -LiteralPath (Join-Path $evidenceDirectory 'docker-image-inspect.json') -Encoding UTF8
$imageId = ($imageInspection | ConvertFrom-Json)[0].Id
$containerId = docker create $ImageName
if ($LASTEXITCODE -ne 0) { throw 'Evidence container creation failed' }
try {
    $buildEvidenceDirectory = Join-Path $evidenceDirectory 'docker-build-evidence'
    New-Item -ItemType Directory -Path $buildEvidenceDirectory -Force | Out-Null
    docker cp "${containerId}:/opt/c101/evidence/." $buildEvidenceDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Build evidence extraction failed' }
} finally {
    docker rm $containerId | Out-Null
}
Invoke-DockerLogged -DockerArguments @('run', '--rm', '--cpuset-cpus', '0,1', '--mount',
    "type=bind,source=$evidenceDirectory,target=/evidence", $ImageName, 'python',
    'scripts/record_c101_linux_lock.py', '--external-root', '/opt/c101/external',
    '--image-id', $imageId, '--output', '/evidence/environment-lock.json') `
    -LogPath (Join-Path $evidenceDirectory 'docker-environment-lock.log')
Write-Host "BUILD COMPLETE: $imageId"
Write-Host 'No solver smoke or full benchmark was started. Send docker-build.log and the final output.'
