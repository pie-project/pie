# pie installer - served at https://pie-project.org/install.ps1
#
#   irm https://pie-project.org/install.ps1 | iex
#   $env:PIE_VERSION = "0.4.0"; irm https://pie-project.org/install.ps1 | iex
#
# Environment overrides:
#   PIE_VERSION       Release tag (default: 0.4.0).
#   PIE_FLAVOR        cuda (the only published Windows build; native CUDA, default).
#   PIE_INSTALL_DIR   Install location for pie.exe (default: %LOCALAPPDATA%\Pie\bin).
#   PIE_REPO          GitHub owner/name (default: pie-project/pie).
#   PIE_DOWNLOAD_BASE Override the asset base URL (default: GitHub releases).
#   PIE_NO_PATH       Do not add PIE_INSTALL_DIR to the user PATH when set.

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

function Write-Heading($Message) {
    Write-Host ""
    Write-Host $Message -ForegroundColor White
}

function Write-Detail($Message) {
    Write-Host "  $Message" -ForegroundColor DarkGray
}

function Write-Step($Message) {
    Write-Host "==> $Message" -ForegroundColor White
}

function Write-Ok($Message) {
    Write-Host "success: $Message" -ForegroundColor Green
}

function Write-Warn($Message) {
    Write-Host "warning: $Message" -ForegroundColor Yellow
}

function Stop-WithError($Message) {
    Write-Host "error: $Message" -ForegroundColor Red
    exit 1
}

function Get-EnvOrDefault($Name, $Default) {
    $value = [Environment]::GetEnvironmentVariable($Name, "Process")
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $Default
    }
    return $value
}

function Test-PathEntry($PathValue, $Entry) {
    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return $false
    }
    foreach ($part in ($PathValue -split ";")) {
        if ($part.TrimEnd("\") -ieq $Entry.TrimEnd("\")) {
            return $true
        }
    }
    return $false
}

function Add-UserPath($Entry) {
    if ([Environment]::GetEnvironmentVariable("PIE_NO_PATH", "Process")) {
        Write-Warn "$Entry was not added to PATH because PIE_NO_PATH is set."
        if (-not (Test-PathEntry $env:Path $Entry)) {
            $env:Path = "$Entry;$env:Path"
        }
        return
    }

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not (Test-PathEntry $userPath $Entry)) {
        if ([string]::IsNullOrWhiteSpace($userPath)) {
            $newPath = $Entry
        } else {
            $newPath = "$userPath;$Entry"
        }
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        Write-Warn "$Entry was added to your user PATH. Open a new terminal to use pie from PATH."
    }

    if (-not (Test-PathEntry $env:Path $Entry)) {
        $env:Path = "$Entry;$env:Path"
    }
}

function Save-Url($Url, $OutFile) {
    $command = Get-Command Invoke-WebRequest
    $params = @{
        Uri = $Url
        OutFile = $OutFile
    }
    if ($command.Parameters.ContainsKey("UseBasicParsing")) {
        $params.UseBasicParsing = $true
    }

    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Invoke-WebRequest @params
            return
        } catch {
            if ($attempt -eq 3) {
                throw
            }
            Start-Sleep -Seconds (2 * $attempt)
        }
    }
}

function Get-AssetName($Flavor) {
    switch ($Flavor.ToLowerInvariant()) {
        "cuda" {
            return "pie-x86_64-windows-cuda.zip"
        }
        default {
            Stop-WithError "no '$Flavor' build for windows/x86_64. The only published Windows build is 'cuda' (native CUDA; requires an NVIDIA GPU)."
        }
    }
}

try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
} catch {
    # Best effort for older Windows PowerShell hosts.
}

if ($env:OS -ne "Windows_NT") {
    Stop-WithError "install.ps1 only supports native Windows. On Linux/macOS, use install.sh."
}

$machineArch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
if ($machineArch -notin @("AMD64", "x86_64")) {
    Stop-WithError "unsupported architecture: $machineArch. Windows ARM64 builds are not published yet."
}

$repo = Get-EnvOrDefault "PIE_REPO" "pie-project/pie"
$version = Get-EnvOrDefault "PIE_VERSION" "0.4.0"
$installDir = Get-EnvOrDefault "PIE_INSTALL_DIR" (Join-Path $env:LOCALAPPDATA "Pie\bin")
$downloadBase = Get-EnvOrDefault "PIE_DOWNLOAD_BASE" "https://github.com/$repo/releases/download/$version"
$flavor = Get-EnvOrDefault "PIE_FLAVOR" "cuda"
$asset = Get-AssetName $flavor
$url = "$downloadBase/$asset"

Write-Heading "Installing Pie"
Write-Detail "Version:  $version"
Write-Detail "Platform: windows/x86_64"
Write-Detail "Flavor:   $flavor"
Write-Detail "Target:   $(Join-Path $installDir "pie.exe")"
if ($env:PIE_VERBOSE) {
    Write-Detail "Asset:    $url"
}

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("pie-install-" + [Guid]::NewGuid().ToString("N"))
$archive = Join-Path $tmp $asset
$extractDir = Join-Path $tmp "extract"

try {
    New-Item -ItemType Directory -Force -Path $tmp, $extractDir | Out-Null

    Write-Step "Downloading release archive"
    try {
        Save-Url $url $archive
    } catch {
        Stop-WithError "download failed for $url. Check PIE_VERSION=$version and PIE_FLAVOR=$flavor, then try again."
    }

    Write-Step "Extracting archive"
    try {
        Expand-Archive -Path $archive -DestinationPath $extractDir -Force
    } catch {
        Stop-WithError "downloaded file is not a valid zip archive: $url"
    }

    $sourceExe = Get-ChildItem -Path $extractDir -Filter "pie.exe" -File -Recurse | Select-Object -First 1
    if (-not $sourceExe) {
        Stop-WithError "archive did not contain a pie.exe binary"
    }

    Write-Step "Installing binary"
    New-Item -ItemType Directory -Force -Path $installDir | Out-Null
    Copy-Item -Force -Path $sourceExe.FullName -Destination (Join-Path $installDir "pie.exe")

    Add-UserPath $installDir

    Write-Ok "Pie was installed to $(Join-Path $installDir "pie.exe")"
    Write-Heading "Next steps"
    Write-Detail "pie --version"
    Write-Detail "pie config init"
    Write-Detail "pie doctor"
} finally {
    if (Test-Path $tmp) {
        Remove-Item -Recurse -Force $tmp
    }
}
