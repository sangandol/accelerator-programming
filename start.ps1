[CmdletBinding()]
param(
  [ValidateRange(1024, 65535)]
  [int]$Port = 43202,
  [switch]$NoBrowser,
  [switch]$Setup
)

# Start the local book reader and keep it attached to this PowerShell process.
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Test-PortBindable([int]$Candidate) {
  $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Candidate)
  try { $listener.Start(); return $true } catch { return $false } finally { try { $listener.Stop() } catch { } }
}

$node = Get-Command node.exe -ErrorAction SilentlyContinue
if ($null -eq $node) { throw "Node.js 20 or newer is required (node.exe not found on PATH)." }

if ($Setup -or -not (Test-Path -LiteralPath ".\node_modules\katex" -PathType Container)) {
  Write-Host "[setup] Installing locked dependencies..."
  & npm.cmd ci --no-fund --no-audit
  if ($LASTEXITCODE -ne 0) { throw "npm ci failed." }
}

if (-not (Test-PortBindable $Port)) { throw "Port $Port is already in use or unavailable." }

$url = "http://127.0.0.1:$Port/book/"
$server = Start-Process -PassThru -NoNewWindow -FilePath $node.Source -ArgumentList @("server.mjs", "--port", "$Port")
try {
  $ready = $false
  for ($attempt = 0; $attempt -lt 40 -and -not $server.HasExited; $attempt++) {
    try {
      $response = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 1 -UseBasicParsing
      if ($response.StatusCode -eq 200) { $ready = $true; break }
    } catch {
      Start-Sleep -Milliseconds 250
    }
  }
  if (-not $ready) { throw "The reader did not become healthy on port $Port." }
  if (-not $NoBrowser) { Start-Process $url }
  Write-Host "[ready] $url  (Ctrl+C to stop)"
  $server.WaitForExit()
} finally {
  if (-not $server.HasExited) {
    Write-Host "[stop] Stopping the reader..."
    Stop-Process -Id $server.Id -Force
  }
}
