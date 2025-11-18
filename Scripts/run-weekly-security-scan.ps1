# PowerShell script to run weekly security scan locally
# This script mimics what the GitHub Actions workflow does

param(
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

Write-Host "🔒 Starting Weekly Security Scan..." -ForegroundColor Cyan
Write-Host ""

# Create reports directory
$ReportsDir = "security-reports-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
New-Item -ItemType Directory -Path $ReportsDir -Force | Out-Null
Write-Host "📁 Reports will be saved to: $ReportsDir" -ForegroundColor Green
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "requirements\dev.txt")) {
    Write-Host "❌ Error: requirements\dev.txt not found" -ForegroundColor Red
    Write-Host "Please run this script from the i-drill directory" -ForegroundColor Yellow
    exit 1
}

# Install security tools if not already installed
Write-Host "📦 Installing security tools..." -ForegroundColor Cyan
try {
    pip install -q bandit[toml] pip-audit safety
    Write-Host "✅ Security tools installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install security tools" -ForegroundColor Red
    exit 1
}
Write-Host ""

# ==================== Bandit Scan ====================
Write-Host "🔍 Running Bandit scan..." -ForegroundColor Cyan
try {
    bandit -r src/ -f json -o "$ReportsDir\bandit-report.json" -ll --exclude src/backend/tests,src/tests 2>&1 | Out-Null
    Write-Host "✅ Bandit scan completed" -ForegroundColor Green
    
    # Console output
    bandit -r src/ -f screen -ll --exclude src/backend/tests,src/tests 2>&1 | Tee-Object -FilePath "$ReportsDir\bandit-console.txt"
} catch {
    Write-Host "⚠️  Bandit found issues (check report)" -ForegroundColor Yellow
    bandit -r src/ -f screen -ll --exclude src/backend/tests,src/tests 2>&1 | Tee-Object -FilePath "$ReportsDir\bandit-console.txt"
}
Write-Host ""

# ==================== pip-audit ====================
Write-Host "🔍 Running pip-audit..." -ForegroundColor Cyan
$RequirementFiles = @(
    "requirements\backend.txt",
    "requirements\ml.txt",
    "requirements\dev.txt"
)

foreach ($reqFile in $RequirementFiles) {
    if (Test-Path $reqFile) {
        $fileName = Split-Path -Leaf $reqFile
        Write-Host "  Checking $fileName..." -ForegroundColor Cyan
        try {
            pip-audit --requirement $reqFile --format json --output "$ReportsDir\pip-audit-$fileName.json" --desc 2>&1 | Out-Null
            pip-audit --requirement $reqFile --desc 2>&1 | Tee-Object -FilePath "$ReportsDir\pip-audit-$fileName-console.txt"
        } catch {
            # Continue on error
        }
    }
}
Write-Host "✅ pip-audit completed" -ForegroundColor Green
Write-Host ""

# ==================== Safety Check ====================
Write-Host "🔍 Running Safety check..." -ForegroundColor Cyan
try {
    $safetyOutput = safety check --file requirements/backend.txt --file requirements/ml.txt --file requirements/dev.txt --json --output "$ReportsDir\safety-report.json" 2>&1
    Write-Host "✅ Safety check passed" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Safety found issues (check report)" -ForegroundColor Yellow
    safety check --file requirements/backend.txt --file requirements/ml.txt --file requirements/dev.txt 2>&1 | Tee-Object -FilePath "$ReportsDir\safety-console.txt"
}
Write-Host ""

# ==================== Generate Summary ====================
Write-Host "📊 Generating summary..." -ForegroundColor Cyan
$SummaryContent = @"
# Security Scan Summary

**Date**: $(Get-Date)
**Reports Directory**: $ReportsDir

## Reports Generated

- ✅ Bandit Report: `bandit-report.json`
- ✅ pip-audit Reports: `pip-audit-*.json`
- ✅ Safety Report: `safety-report.json`

## View Reports

``````powershell
# View JSON reports
Get-Content $ReportsDir\*.json

# View console outputs
Get-Content $ReportsDir\*-console.txt

# View this summary
Get-Content $ReportsDir\SUMMARY.md
``````

## Next Steps

1. Review all reports
2. Fix Critical and High severity issues
3. Update dependencies if needed
4. Re-run scan after fixes
"@

$SummaryContent | Out-File -FilePath "$ReportsDir\SUMMARY.md" -Encoding UTF8
Write-Host "✅ Summary generated: $ReportsDir\SUMMARY.md" -ForegroundColor Green
Write-Host ""

# ==================== Final Summary ====================
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "✅ Weekly Security Scan Completed!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
Write-Host "📁 Reports saved to: $ReportsDir" -ForegroundColor Green
Write-Host "📄 Summary: $ReportsDir\SUMMARY.md" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Review reports in $ReportsDir"
Write-Host "  2. Fix Critical/High severity issues"
Write-Host "  3. Update dependencies if needed"
Write-Host ""

