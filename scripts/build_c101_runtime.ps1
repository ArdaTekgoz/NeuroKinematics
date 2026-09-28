[CmdletBinding()]
param(
    [ValidateLength(1, 48)]
    [ValidatePattern('(?-i)^[a-z0-9]+(?:-[a-z0-9]+)*$')]
    [string]$BuildName = 'runtime-build-v1',

    [switch]$ResumeLock
)

$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$evidenceRoot = Join-Path $taskRoot 'experiments\C1-01'
$baseLockPath = Join-Path $evidenceRoot 'environment-lock.json'
$buildDirectory = Join-Path $evidenceRoot $BuildName
$contextDirectory = Join-Path $buildDirectory 'inputs'
$imageName = "neurokinematics-c101:$BuildName"
$baseTag = "neurokinematics-c101:$BuildName-base"
$dockerExecutable = (Get-Command docker.exe -CommandType Application -ErrorAction Stop | Select-Object -First 1).Path
if ((Test-Path -LiteralPath $buildDirectory) -and -not $ResumeLock) {
    throw "Build evidence already exists; choose a new -BuildName: $buildDirectory"
}
$baseLock = Get-Content -LiteralPath $baseLockPath -Raw | ConvertFrom-Json
$baseImageId = [string]$baseLock.image_id
if ($baseImageId -cnotmatch '^sha256:[0-9a-f]{64}$') { throw 'Invalid immutable base image ID.' }
if ((Get-FileHash -LiteralPath (Join-Path $taskRoot 'pixi.lock') -Algorithm SHA256).Hash.ToLowerInvariant() -cne $baseLock.pixi_lock_sha256) {
    throw 'Host pixi.lock differs from the audited base; dependency changes are outside this rebuild.'
}

function Invoke-DockerCapture {
    param([string[]]$DockerArguments)
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $lines = & $dockerExecutable @DockerArguments
        $exitCode = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previousPreference }
    if ($exitCode -ne 0) { throw "Docker command failed (exit $exitCode): $($DockerArguments[0])" }
    return $lines
}

function Invoke-DockerLogged {
    param([string[]]$DockerArguments, [string]$LogPath)
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $dockerExecutable @DockerArguments 2>&1 | Tee-Object -FilePath $LogPath -ErrorAction Stop
        $exitCode = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previousPreference }
    if ($exitCode -ne 0) { throw "Docker command failed (exit $exitCode); see $LogPath" }
}

function Write-Json {
    param([object]$Value, [string]$Path)
    $text = ($Value | ConvertTo-Json -Depth 30) + [Environment]::NewLine
    [IO.File]::WriteAllText($Path, $text, (New-Object Text.UTF8Encoding($false)))
}

function Copy-SnapshotTree {
    param([string]$RelativePath, [switch]$PythonOnly)
    $sourceRoot = (Get-Item -LiteralPath (Join-Path $taskRoot $RelativePath) -Force).FullName.TrimEnd('\')
    $items = @(Get-Item -LiteralPath $sourceRoot -Force) + @(Get-ChildItem -LiteralPath $sourceRoot -Recurse -Force)
    foreach ($item in $items) {
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Linked build input is unsupported: $($item.FullName)"
        }
    }
    foreach ($item in $items) {
        if ($item.PSIsContainer -or ($PythonOnly -and $item.Extension -cne '.py')) { continue }
        $relativeFile = Join-Path $RelativePath $item.FullName.Substring($sourceRoot.Length + 1)
        $destination = Join-Path $contextDirectory $relativeFile
        New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
        Copy-Item -LiteralPath $item.FullName -Destination $destination -ErrorAction Stop
    }
}

function Complete-RuntimeLock {
    param([string]$ImageId)
    $finalLock = Join-Path $buildDirectory 'environment-lock.json'
    if (Test-Path -LiteralPath $finalLock) { throw 'Published environment lock already exists; refusing to overwrite.' }
    $attempt = '{0}-{1}' -f (Get-Date -Format 'yyyyMMdd-HHmmss-fff'), ([guid]::NewGuid().ToString('N'))
    $candidateName = "environment-lock.candidate-$attempt.json"
    # The inherited entrypoint already supplies --locked. Suppress environment
    # reinstall when source snapshots changed; use the installed audited env.
    Invoke-DockerLogged -DockerArguments @('run', '--rm', '--network', 'none', '--cpuset-cpus', '0,1',
        '-e', 'PIXI_NO_INSTALL=true', '--mount', "type=bind,source=$buildDirectory,target=/evidence",
        $ImageId, 'python', 'scripts/record_c101_linux_lock.py', '--external-root', '/opt/c101/external',
        '--image-id', $ImageId, '--output', "/evidence/$candidateName") `
        -LogPath (Join-Path $buildDirectory "docker-environment-lock-$attempt.log")
    Invoke-DockerLogged -DockerArguments @('run', '--rm', '--network', 'none', '--cpuset-cpus', '0,1',
        '-e', 'PIXI_NO_INSTALL=true', '--mount', "type=bind,source=$buildDirectory,target=/evidence,readonly",
        $ImageId, 'python', '/opt/c101/runtime/audit_c101_runtime_build.py',
        '--base-lock', '/opt/c101/runtime/base-environment-lock.json',
        '--recorded-lock', "/evidence/$candidateName", '--input-manifest', '/opt/c101/runtime/input-manifest.json') `
        -LogPath (Join-Path $buildDirectory "dependency-closure-audit-$attempt.log")
    Rename-Item -LiteralPath (Join-Path $buildDirectory $candidateName) -NewName 'environment-lock.json'
    Write-Host "RUNTIME BUILD COMPLETE: $ImageId"
    Write-Host "New environment lock: $finalLock"
    Write-Host 'Send the final output before running the prepare stage. No smoke or benchmark was started.'
}

if ($ResumeLock) {
    $savedPath = Join-Path $buildDirectory 'docker-image-inspect.json'
    $manifestPath = Join-Path $buildDirectory 'input-manifest.json'
    if (-not (Test-Path -LiteralPath $savedPath -PathType Leaf) -or -not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw 'ResumeLock requires an existing completed image build and input manifest.'
    }
    $savedImages = @(Get-Content -LiteralPath $savedPath -Raw | ConvertFrom-Json)
    if ($savedImages.Count -ne 1 -or $savedImages[0].Id -cnotmatch '^sha256:[0-9a-f]{64}$') { throw 'Invalid saved image identity.' }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($manifest.base_image_id -cne $baseImageId -or
        $manifest.base_environment_lock_sha256 -cne (Get-FileHash -LiteralPath $baseLockPath -Algorithm SHA256).Hash.ToLowerInvariant()) {
        throw 'Base environment lock changed since the recorded build.'
    }
    $inspection = Invoke-DockerCapture -DockerArguments @('image', 'inspect', $savedImages[0].Id)
    $images = @($inspection | ConvertFrom-Json)
    if ($images.Count -ne 1 -or $images[0].Id -cne $savedImages[0].Id -or
        $images[0].Os -cne 'linux' -or $images[0].Architecture -cne 'amd64' -or
        $images[0].Config.Labels.'org.neurokinematics.c101.base-image-id' -cne $baseImageId) {
        throw 'Resume image identity or base provenance mismatch.'
    }
    Write-Host "Resuming environment lock for existing image: $($images[0].Id)"
    Complete-RuntimeLock -ImageId $images[0].Id
    exit 0
}

$baseInspection = Invoke-DockerCapture -DockerArguments @('image', 'inspect', $baseImageId)
$baseImages = @($baseInspection | ConvertFrom-Json)
if ($baseImages.Count -ne 1 -or $baseImages[0].Id -cne $baseImageId -or
    $baseImages[0].Os -cne 'linux' -or $baseImages[0].Architecture -cne 'amd64') {
    throw 'The audited Linux amd64 base image is not available locally.'
}
$existingTarget = @(Invoke-DockerCapture -DockerArguments @('image', 'ls', '--quiet', '--no-trunc', '--filter', "reference=$imageName"))
if ($existingTarget.Count -ne 0) { throw "Image tag already exists; choose a new -BuildName: $imageName" }
$existingBase = @(Invoke-DockerCapture -DockerArguments @('image', 'ls', '--quiet', '--no-trunc', '--filter', "reference=$baseTag"))
if ($existingBase.Count -gt 0 -and ($existingBase.Count -ne 1 -or $existingBase[0] -cne $baseImageId)) {
    throw "Base alias already identifies another image: $baseTag"
}
New-Item -ItemType Directory -Path $buildDirectory -ErrorAction Stop | Out-Null
New-Item -ItemType Directory -Path $contextDirectory -ErrorAction Stop | Out-Null
$baseInspection | Set-Content -LiteralPath (Join-Path $buildDirectory 'base-image-inspect.json') -Encoding UTF8
Copy-Item -LiteralPath $baseLockPath -Destination (Join-Path $contextDirectory 'base-environment-lock.json')
Copy-Item -LiteralPath (Join-Path $evidenceRoot 'Dockerfile.runtime') -Destination (Join-Path $contextDirectory 'Dockerfile')
Copy-Item -LiteralPath (Join-Path $evidenceRoot 'Dockerfile.runtime.dockerignore') -Destination (Join-Path $contextDirectory '.dockerignore')
Copy-SnapshotTree -RelativePath 'src' -PythonOnly
Copy-SnapshotTree -RelativePath 'tests' -PythonOnly
Copy-SnapshotTree -RelativePath 'scripts' -PythonOnly
Copy-SnapshotTree -RelativePath 'ros2_ws\src\c101_moveit_worker'
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $buildDirectory 'build_c101_runtime.ps1')
$inputHashes = [ordered]@{}
foreach ($inputFile in (Get-ChildItem -LiteralPath $contextDirectory -Recurse -File -Force | Sort-Object FullName)) {
    $relativeFile = $inputFile.FullName.Substring($contextDirectory.Length + 1).Replace('\', '/')
    $inputHashes[$relativeFile] = (Get-FileHash -LiteralPath $inputFile.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
}
$manifest = [ordered]@{
    schema_version = '1.0.0'; task = 'C1-01'; scope = 'local worker rebuild with inherited dependencies'
    recorded_utc = [DateTime]::UtcNow.ToString('o'); base_image_id = $baseImageId
    base_environment_lock_sha256 = $inputHashes['base-environment-lock.json']
    launcher_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    files = $inputHashes
}
Write-Json -Value $manifest -Path (Join-Path $contextDirectory 'input-manifest.json')
Copy-Item -LiteralPath (Join-Path $contextDirectory 'input-manifest.json') -Destination (Join-Path $buildDirectory 'input-manifest.json')

# A local tag avoids BuildKit interpreting an image ID as a remote repository.
# The immutable ID is inspected first, retained in the image label and manifest.
Invoke-DockerCapture -DockerArguments @('image', 'tag', $baseImageId, $baseTag) | Out-Null
Invoke-DockerLogged -DockerArguments @('build', '--platform', 'linux/amd64', '--pull=false', '--network', 'none',
    '--progress', 'plain', '--build-arg', "C101_BASE_IMAGE=$baseTag", '--build-arg', "C101_BASE_IMAGE_ID=$baseImageId",
    '-f', (Join-Path $contextDirectory 'Dockerfile'), '-t', $imageName, $contextDirectory) `
    -LogPath (Join-Path $buildDirectory 'docker-build.log')
$imageInspection = Invoke-DockerCapture -DockerArguments @('image', 'inspect', $imageName)
$imageInspection | Set-Content -LiteralPath (Join-Path $buildDirectory 'docker-image-inspect.json') -Encoding UTF8
$image = @($imageInspection | ConvertFrom-Json)[0]
if ($image.Id -cnotmatch '^sha256:[0-9a-f]{64}$' -or $image.Os -cne 'linux' -or $image.Architecture -cne 'amd64' -or
    $image.Config.Labels.'org.neurokinematics.c101.base-image-id' -cne $baseImageId) { throw 'Built image identity mismatch.' }
Complete-RuntimeLock -ImageId $image.Id
