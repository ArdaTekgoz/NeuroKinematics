$ErrorActionPreference = 'Stop'
$c101Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$c101Scratch = Join-Path $c101Root ('tmp\session-launcher-check-' + [guid]::NewGuid().ToString('N'))
$c101Fixture = Join-Path $c101Scratch 'repo'
$c101Bin = Join-Path $c101Scratch 'bin'
foreach ($c101Relative in @('scripts', 'src', 'tests', 'experiments\C1-01\runtime-build-v1')) {
    New-Item -ItemType Directory -Path (Join-Path $c101Fixture $c101Relative) -Force | Out-Null
}
New-Item -ItemType Directory -Path $c101Bin | Out-Null
Copy-Item -LiteralPath (Join-Path $c101Root 'scripts/run_c101_session.ps1') -Destination (Join-Path $c101Fixture 'scripts/run_c101_session.ps1')
Copy-Item -LiteralPath (Join-Path $c101Root 'scripts/c101_session.py') -Destination (Join-Path $c101Fixture 'scripts/c101_session.py')
$c101FixtureLock = Join-Path $c101Fixture 'experiments\C1-01\runtime-build-v1\environment-lock.json'
Copy-Item -LiteralPath (Join-Path $c101Root 'experiments/C1-01/runtime-build-v1/environment-lock.json') -Destination $c101FixtureLock
'# Snapshot source fixture' | Set-Content -LiteralPath (Join-Path $c101Fixture 'src/example.py')
'# Snapshot test fixture' | Set-Content -LiteralPath (Join-Path $c101Fixture 'tests/example.py')
$c101Record = Get-Content -LiteralPath $c101FixtureLock -Raw | ConvertFrom-Json
$c101FakeInspect = Join-Path $c101Scratch 'image.json'
ConvertTo-Json -InputObject @([pscustomobject]@{ Id=$c101Record.image_id; Os='linux'; Architecture='amd64' }) | Set-Content -LiteralPath $c101FakeInspect -Encoding UTF8
$c101FakeArgs = Join-Path $c101Scratch 'run-args.txt'
$c101FakeProgram = @'
using System;
using System.IO;
public class C101FakeDocker {
    public static int Main(string[] args) {
        if (args.Length == 3 && args[0] == "image" && args[1] == "inspect") {
            Console.WriteLine(File.ReadAllText(Environment.GetEnvironmentVariable("C101_FAKE_INSPECTION")));
            return 0;
        }
        if (args.Length > 0 && args[0] == "run") {
            File.WriteAllLines(Environment.GetEnvironmentVariable("C101_FAKE_ARGS"), args);
            Console.WriteLine("MOCK DOCKER: no container executed");
            return Environment.GetEnvironmentVariable("C101_FAKE_FAIL") == "true" ? 19 : 0;
        }
        Console.Error.WriteLine("Unexpected mock Docker command");
        return 88;
    }
}
'@
Add-Type -TypeDefinition $c101FakeProgram -OutputAssembly (Join-Path $c101Bin 'docker.exe') -OutputType ConsoleApplication
$c101OriginalPath = $env:PATH
$env:PATH = $c101Bin + ';' + $env:PATH
$env:C101_FAKE_INSPECTION = $c101FakeInspect
$env:C101_FAKE_ARGS = $c101FakeArgs
$env:C101_FAKE_FAIL = 'false'
$c101Launcher = Join-Path $c101Fixture 'scripts/run_c101_session.ps1'
$c101Results = @()
try {
    foreach ($c101Case in @('relative', 'absolute')) {
        if ($c101Case -eq 'relative') { & $c101Launcher -Stage prepare -SessionName "mock-$c101Case" }
        else { & $c101Launcher -Stage prepare -SessionName "mock-$c101Case" -EnvironmentLock $c101FixtureLock }
        $c101Args = Get-Content -LiteralPath $c101FakeArgs
        if ($c101Args -notcontains $c101Record.image_id -or $c101Args -notcontains 'PIXI_NO_INSTALL=true') { throw 'Image ID or no-install argument missing.' }
        $c101MemoryIndex = [array]::IndexOf($c101Args, '--memory')
        $c101SwapIndex = [array]::IndexOf($c101Args, '--memory-swap')
        if ($c101MemoryIndex -lt 0 -or $c101SwapIndex -lt 0 -or $c101Args[$c101MemoryIndex + 1] -cne '8g' -or $c101Args[$c101SwapIndex + 1] -cne '8g') { throw 'Fixed 8 GiB / zero swap policy missing.' }
        $c101CopiedLock = Join-Path $c101Fixture "experiments\C1-01\mock-$c101Case\base-environment-lock.json"
        if ((Get-FileHash -LiteralPath $c101CopiedLock).Hash -ne (Get-FileHash -LiteralPath $c101FixtureLock).Hash) { throw 'Snapshot lock changed.' }
        $c101Results += [pscustomobject]@{ scenario="prepare with $c101Case lock path"; status='PASS' }
    }
    & $c101Launcher -Stage smoke -SessionName mock-relative
    if ((Get-Content -LiteralPath $c101FakeArgs) -notcontains $c101Record.image_id) { throw 'Later stage lost image ID.' }
    $c101Results += [pscustomobject]@{ scenario='later stage reads session lock copy'; status='PASS' }
    $env:C101_FAKE_FAIL = 'true'
    $c101Caught = $false
    try { & $c101Launcher -Stage pilot -SessionName mock-relative } catch {
        if ($_.Exception.Message -notmatch 'Docker exit 19') { throw }
        $c101Caught = $true
    }
    if (-not $c101Caught) { throw 'Docker failure was reported as success.' }
    $c101Results += [pscustomobject]@{ scenario='native failure propagates'; status='PASS' }
} finally {
    $env:PATH = $c101OriginalPath
    Remove-Item Env:C101_FAKE_INSPECTION, Env:C101_FAKE_ARGS, Env:C101_FAKE_FAIL -ErrorAction SilentlyContinue
}
$c101Evidence = [pscustomobject]@{ recorded_utc=[DateTime]::UtcNow.ToString('o'); powershell_version=$PSVersionTable.PSVersion.ToString(); real_docker_invoked=$false; checks=$c101Results }
$c101Evidence | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $c101Root 'experiments/C1-01/session-launcher-flow-check.json') -Encoding UTF8
$c101Evidence | ConvertTo-Json -Depth 5
