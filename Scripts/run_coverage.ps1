# PowerShell script to run tests with coverage report

Write-Host "🧪 Running tests with coverage..." -ForegroundColor Cyan

# Change to backend directory
$backendDir = Join-Path $PSScriptRoot "..\src\backend"
Set-Location $backendDir

# Run tests with coverage
pytest tests/ `
    -v `
    --cov=. `
    --cov-report=html `
    --cov-report=term-missing `
    --cov-report=xml `
    --cov-branch `
    --cov-fail-under=70

# Check if coverage report was generated
if (Test-Path "htmlcov") {
    Write-Host ""
    Write-Host "✅ Coverage report generated in htmlcov/index.html" -ForegroundColor Green
    Write-Host "📊 Open the report with: Start-Process htmlcov\index.html" -ForegroundColor Yellow
} else {
    Write-Host "⚠️  Coverage report not generated" -ForegroundColor Red
    exit 1
}

