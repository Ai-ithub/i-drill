# PowerShell script to fix commit messages using git filter-branch
# This script will change commit messages for specific commits

Write-Host "Starting commit message fix..." -ForegroundColor Green
Write-Host ""

# Set environment variable
$env:FILTER_BRANCH_SQUELCH_WARNING = "1"

# Get the full path to the current directory
$currentDir = (Get-Location).Path
$scriptPath = Join-Path $currentDir "commit_msg_filter.ps1"

# Create the filter script
$filterScript = @'
$commitHash = $env:GIT_COMMIT

switch ($commitHash) {
    "696ad6000c518b4699f22de2624cea803451675d" {
        "chore: Improve security configurations and phase zero maintenance infrastructure`n`n- Add security headers and middleware improvements`n- Enhance health check endpoints`n- Update docker-compose configuration`n- Improve Kafka service reliability`n- Update setup guide documentation"
    }
    "55c0e0d298dbb9cb20cb534bcc1014ae0584c71c" {
        "feat: Add RL system management and autonomous mode`n`n- Implement RL control page with system management interface`n- Add RL service with autonomous mode support`n- Create RL API routes for system control`n- Add comprehensive RL API tests`n- Enhance RL service with advanced control features"
    }
    "8a934b16a8fb02582a97458efe43aab8f67ba866" {
        "feat: Complete backend-dashboard integration and automated CI`n`n- Enhanced frontend pages (HistoricalData, Maintenance, Predictions)`n- Improved backend services (maintenance, prediction, MLflow)`n- Added comprehensive API tests for maintenance and predictions`n- Updated CI/CD workflows`n- WebSocket service improvements"
    }
    default {
        $input
    }
}
'@

$filterScript | Out-File -FilePath $scriptPath -Encoding UTF8

Write-Host "Created filter script: $scriptPath" -ForegroundColor Cyan
Write-Host ""

# Run filter-branch
Write-Host "Running git filter-branch..." -ForegroundColor Yellow
$result = git filter-branch -f --msg-filter "powershell -File `"$scriptPath`"" -- --all 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "Successfully changed commit messages!" -ForegroundColor Green
} else {
    Write-Host "Error occurred:" -ForegroundColor Red
    Write-Host $result
}

# Clean up
Remove-Item $scriptPath -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Done! Check the results with: git log --oneline | Select-String '55c0e0d|696ad60|8a934b1'" -ForegroundColor Cyan

