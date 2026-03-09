[CmdletBinding()]
param(
    [string]$NasHost = "192.168.2.10",
    [string]$NasUser = "VKotenok",
    [string]$ProjectPath = "/volume1/docker/anki-csv-builder/app",
    [string]$Branch = "dev",
    [string]$SshKeyPath = "~/.ssh/id_ed25519",
    [int]$SshPort = 22,
    [switch]$SkipLocalHttpChecks
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Prefix,
        [Parameter(Mandatory = $true)][string]$Message
    )
    Write-Host "[$Prefix] $Message"
}

function Expand-UserPath {
    param([Parameter(Mandatory = $true)][string]$PathValue)

    $expanded = [Environment]::ExpandEnvironmentVariables($PathValue)
    if ($expanded -eq "~") {
        $expanded = $HOME
    } elseif ($expanded.StartsWith("~/") -or $expanded.StartsWith("~\")) {
        $expanded = Join-Path -Path $HOME -ChildPath $expanded.Substring(2)
    }
    return [System.IO.Path]::GetFullPath($expanded)
}

function ConvertTo-BashSingleQuoted {
    param([Parameter(Mandatory = $true)][string]$Value)
    return "'" + ($Value -replace "'", "'\''") + "'"
}

function Invoke-SshChecked {
    param(
        [Parameter(Mandatory = $true)][string[]]$Args,
        [string]$InputText
    )

    if ($null -ne $InputText) {
        $InputText | & ssh @Args
    } else {
        & ssh @Args
    }
    if ($LASTEXITCODE -ne 0) {
        throw "ssh command failed with exit code $LASTEXITCODE"
    }
}

function Wait-ApiHealth {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [int]$TimeoutSeconds = 60,
        [int]$IntervalSeconds = 3
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        try {
            $response = Invoke-WebRequest -Uri $Url -Method Get -TimeoutSec 8
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 400 -and $response.Content -match "ok|healthy") {
                return $true
            }
        } catch {
            # Keep retrying until timeout.
        }
        Start-Sleep -Seconds $IntervalSeconds
    } while ((Get-Date) -lt $deadline)

    return $false
}

function Wait-WebUi {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [int]$TimeoutSeconds = 60,
        [int]$IntervalSeconds = 3
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        try {
            $response = Invoke-WebRequest -Uri $Url -Method Get -TimeoutSec 8
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 400) {
                return $true
            }
        } catch {
            # Keep retrying until timeout.
        }
        Start-Sleep -Seconds $IntervalSeconds
    } while ((Get-Date) -lt $deadline)

    return $false
}

try {
    Write-Step -Prefix "preflight" -Message "Checking local dependencies..."
    $sshCmd = Get-Command ssh -ErrorAction SilentlyContinue
    if (-not $sshCmd) {
        throw "OpenSSH client (ssh) not found in PATH."
    }

    $resolvedKeyPath = Expand-UserPath -PathValue $SshKeyPath
    if (-not (Test-Path -LiteralPath $resolvedKeyPath)) {
        throw "SSH key not found: $resolvedKeyPath"
    }

    $sshCommonArgs = @(
        "-p", "$SshPort",
        "-i", "$resolvedKeyPath",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=8",
        "-o", "StrictHostKeyChecking=accept-new"
    )

    Write-Step -Prefix "preflight" -Message "Testing SSH connectivity to $NasUser@$NasHost..."
    Invoke-SshChecked -Args ($sshCommonArgs + @("$NasUser@$NasHost", "echo '[preflight] ssh-ok'"))

    $projectPathEscaped = ConvertTo-BashSingleQuoted -Value $ProjectPath
    $branchEscaped = ConvertTo-BashSingleQuoted -Value $Branch
    $nasHostEscaped = ConvertTo-BashSingleQuoted -Value $NasHost

    $remoteScript = @"
set -euo pipefail

PROJECT_PATH=$projectPathEscaped
BRANCH=$branchEscaped
NAS_HOST=$nasHostEscaped

echo "[git] project path: \$PROJECT_PATH"
if [[ ! -d "\$PROJECT_PATH" ]]; then
  echo "[git] ERROR: project path not found: \$PROJECT_PATH" >&2
  exit 1
fi
cd "\$PROJECT_PATH"

if [[ ! -d .git ]]; then
  echo "[git] ERROR: \$PROJECT_PATH is not a git checkout." >&2
  exit 1
fi

echo "[git] fetch --prune origin"
git fetch --prune origin

echo "[git] checkout \$BRANCH"
git checkout "\$BRANCH"

echo "[git] pull --ff-only origin \$BRANCH"
git pull --ff-only origin "\$BRANCH"

echo "[smoke] validate env"
bash deploy/synology/scripts/validate_env.sh

echo "[compose] docker compose up -d --build"
docker compose -f deploy/synology/docker-compose.synology.yml --env-file deploy/synology/.env up -d --build

echo "[smoke] smoke.sh \$NAS_HOST"
bash deploy/synology/scripts/smoke.sh "\$NAS_HOST"

echo "[compose] docker compose ps"
docker compose -f deploy/synology/docker-compose.synology.yml --env-file deploy/synology/.env ps

echo "[summary] deployed commit: \$(git rev-parse --short HEAD)"
"@

    Write-Step -Prefix "git" -Message "Running remote deploy pipeline..."
    Invoke-SshChecked -Args ($sshCommonArgs + @("$NasUser@$NasHost", "bash -seuo pipefail")) -InputText $remoteScript

    if (-not $SkipLocalHttpChecks) {
        $apiUrl = "http://$NasHost`:8000/health"
        $webUrl = "http://$NasHost`:5173/"

        Write-Step -Prefix "health" -Message "Checking API health ($apiUrl) with retry..."
        if (-not (Wait-ApiHealth -Url $apiUrl -TimeoutSeconds 60 -IntervalSeconds 3)) {
            throw "API health check failed for $apiUrl after retry window."
        }

        Write-Step -Prefix "health" -Message "Checking Web UI ($webUrl) with retry..."
        if (-not (Wait-WebUi -Url $webUrl -TimeoutSeconds 60 -IntervalSeconds 3)) {
            throw "Web UI check failed for $webUrl after retry window."
        }
        Write-Step -Prefix "health" -Message "Local HTTP checks passed."
    } else {
        Write-Step -Prefix "health" -Message "Skipping local HTTP checks by flag."
    }

    Write-Step -Prefix "summary" -Message "Deploy pipeline completed successfully."
    exit 0
}
catch {
    Write-Error $_
    exit 1
}
