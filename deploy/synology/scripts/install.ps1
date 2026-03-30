<#
.SYNOPSIS
  Remote Synology deploy from Windows PowerShell over SSH.

.DESCRIPTION
  Syncs local repo to NAS (excluding deploy/synology/.env) and runs
  deploy/synology/scripts/install.sh on NAS.
#>
[CmdletBinding()]
param(
  [string]$Nas = "VKotenok@192.168.2.10",
  [string]$LocalRepoPath = "D:\github\anki\dev\anki-csv-builder",
  [string]$RemoteAppPath = "/volume1/docker/anki-csv-builder/app",
  [switch]$SkipSync,
  [switch]$NoBuild,
  [switch]$SkipSmoke,
  [string]$SmokeHost = "",
  [switch]$ForceEnv
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Quote-Bash([string]$Value) {
  $replacement = "'" + [char]34 + "'" + [char]34 + "'"
  $escaped = $Value.Replace("'", $replacement)
  return "'" + $escaped + "'"
}

function Assert-LastExit([string]$StepName) {
  if ($LASTEXITCODE -ne 0) {
    throw "$StepName failed with exit code $LASTEXITCODE"
  }
}

if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
  throw "ssh command is not available in PATH"
}

if (-not $SkipSync) {
  if (-not (Get-Command tar -ErrorAction SilentlyContinue)) {
    throw "tar command is not available in PATH"
  }
  if (-not (Test-Path -LiteralPath $LocalRepoPath)) {
    throw "Local repo path not found: $LocalRepoPath"
  }
}

$remotePathQuoted = Quote-Bash $RemoteAppPath

Write-Host "[1/3] Ensure remote app folder exists: $RemoteAppPath"
& ssh -T $Nas "mkdir -p $remotePathQuoted"
Assert-LastExit "Create remote folder"

if (-not $SkipSync) {
  Write-Host "[2/3] Sync local repo to NAS (preserve remote deploy/synology/.env)"
  $remoteExtract = @'
tmp_err="/tmp/anki_sync_tar_$$.log";
tar -C {0} --no-same-owner --no-same-permissions --no-overwrite-dir -xf - 2>"$tmp_err";
rc=$?;
if [ $rc -ne 0 ] && grep -Evq 'Cannot change mode to .*Operation not permitted|Exiting with failure status due to previous errors' "$tmp_err"; then
  cat "$tmp_err" >&2;
  rm -f "$tmp_err";
  exit $rc;
fi;
cat "$tmp_err" >&2 || true;
rm -f "$tmp_err"
'@.Trim() -f $remotePathQuoted
  $remoteExtract = $remoteExtract -replace "`r?`n", " "
  $cmdSync = @(
    'tar'
    '-C', ('"{0}"' -f $LocalRepoPath)
    '--format=ustar'
    '--exclude', '".git"'
    '--exclude', '".venv"'
    '--exclude', '"venv"'
    '--exclude', '"node_modules"'
    '--exclude', '"web/node_modules"'
    '--exclude', '"deploy/synology/.env"'
    '-cf', '-', '.'
    '|'
    'ssh', '-T', $Nas, ('"{0}"' -f $remoteExtract.Replace('"', '\"'))
  ) -join ' '
  & cmd.exe /d /c $cmdSync
  Assert-LastExit "Sync project files"
} else {
  Write-Host "[2/3] Skip sync requested"
}

$remoteInstallArgs = New-Object System.Collections.Generic.List[string]
if ($NoBuild) { $remoteInstallArgs.Add("--no-build") }
if ($SkipSmoke) { $remoteInstallArgs.Add("--skip-smoke") }
if ($ForceEnv) { $remoteInstallArgs.Add("--force-env") }
if (-not [string]::IsNullOrWhiteSpace($SmokeHost)) {
  $remoteInstallArgs.Add("--smoke-host $(Quote-Bash $SmokeHost)")
}

$remoteInstallCmd = "bash deploy/synology/scripts/install.sh"
if ($remoteInstallArgs.Count -gt 0) {
  $remoteInstallCmd += " " + ($remoteInstallArgs -join " ")
}

Write-Host "[3/3] Run remote install script"
$remoteInstallRunCmd = "cd $remotePathQuoted && $remoteInstallCmd"
$remoteInstallSudoCmd = "cd $remotePathQuoted && sudo -H $remoteInstallCmd"

$plainOutput = @()
$plainExit = 1
try {
  $plainOutput = & ssh -T $Nas $remoteInstallRunCmd 2>&1
  $plainExit = if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } else { 0 }
} catch {
  $plainOutput += ($_ | Out-String)
  $plainExit = if (($null -ne $LASTEXITCODE) -and ([int]$LASTEXITCODE -ne 0)) { [int]$LASTEXITCODE } else { 1 }
}
if ($plainOutput) {
  $plainOutput | ForEach-Object { Write-Host $_ }
}

if ($plainExit -ne 0) {
  $plainText = ($plainOutput | Out-String)
  $isDockerSocketPermission = $plainText -match "permission denied while trying to connect to the Docker daemon socket" -or `
                              $plainText -match "dial unix /var/run/docker.sock: connect: permission denied"

  if ($isDockerSocketPermission) {
    Write-Host "Docker socket permission denied; retrying with sudo..."
    $sudoOutput = @()
    $sudoExit = 1
    try {
      $sudoOutput = & ssh -tt $Nas $remoteInstallSudoCmd 2>&1
      $sudoExit = if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } else { 0 }
    } catch {
      $sudoOutput += ($_ | Out-String)
      $sudoExit = if (($null -ne $LASTEXITCODE) -and ([int]$LASTEXITCODE -ne 0)) { [int]$LASTEXITCODE } else { 1 }
    }
    if ($sudoOutput) {
      $sudoOutput | ForEach-Object { Write-Host $_ }
    }
    if ($sudoExit -ne 0) {
      throw "Remote install (sudo) failed with exit code $sudoExit"
    }
  } else {
    throw "Remote install failed with exit code $plainExit"
  }
}

Write-Host "Remote install completed."
