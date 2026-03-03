# Helper script to activate venv and run the app (PowerShell)
$venv = Join-Path -Path $PSScriptRoot -ChildPath '.venv\Scripts\Activate.ps1'
if (Test-Path $venv) {
    . $venv
} else {
    Write-Host "Virtual environment activate script not found at $venv" -ForegroundColor Yellow
}

python .\app.py
