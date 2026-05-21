param(
    [string]$SecretPath = "$env:USERPROFILE\.tsgnn\hpc_key_passphrase.dpapi"
)

$ErrorActionPreference = "Stop"

$dir = Split-Path -Parent $SecretPath
if (!(Test-Path -LiteralPath $dir)) {
    New-Item -ItemType Directory -Path $dir | Out-Null
}

Write-Host "This stores the Bocconi SSH key passphrase encrypted with Windows DPAPI." -ForegroundColor Cyan
Write-Host "It is readable only by this Windows user on this machine." -ForegroundColor Cyan
$secret = Read-Host "SSH key passphrase / HPC password" -AsSecureString
$secret | ConvertFrom-SecureString | Set-Content -LiteralPath $SecretPath -Encoding ASCII

Write-Host "Saved encrypted secret to $SecretPath" -ForegroundColor Green
