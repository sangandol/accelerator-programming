[CmdletBinding()]
param(
  [ValidateRange(1024, 65535)]
  [int]$Port = 43202,
  [switch]$NoBrowser,
  [switch]$Setup,
  [switch]$Tailscale
)

# Start the local book reader and keep it attached to this PowerShell process.
# -Tailscale also listens on this machine's Tailscale address, so only your own Tailscale devices
# can read the book; it never listens on the Wi-Fi or Ethernet addresses.
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$hosts = @("127.0.0.1")
if ($Tailscale) {
  $cli = Get-Command tailscale.exe -ErrorAction SilentlyContinue
  $cliPath = if ($cli) { $cli.Source } else { "C:\Program Files\Tailscale\tailscale.exe" }
  if (-not (Test-Path -LiteralPath $cliPath)) { throw "Tailscale is not installed (tailscale.exe not found)." }
  $tsAddress = try { "$(@(& $cliPath ip -4 2>$null)[0])".Trim() } catch { "" }
  if ($tsAddress -notmatch '^100\.\d+\.\d+\.\d+$') { throw "Tailscale is not connected (sign in from the Tailscale tray icon)." }
  $hosts += $tsAddress
}

function Test-PortBindable([int]$Candidate) {
  foreach ($address in $hosts) {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse($address), $Candidate)
    try { $listener.Start() } catch { return $false } finally { try { $listener.Stop() } catch { } }
  }
  return $true
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
$server = Start-Process -PassThru -NoNewWindow -FilePath $node.Source -ArgumentList @("server.mjs", "--port", "$Port", "--host", ($hosts -join ","))
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
  if ($Tailscale) { Write-Host "[tailscale] http://$($hosts[1]):$Port/book/  (only your Tailscale devices)" }
  $server.WaitForExit()
} finally {
  if (-not $server.HasExited) {
    Write-Host "[stop] Stopping the reader..."
    Stop-Process -Id $server.Id -Force
  }
}
