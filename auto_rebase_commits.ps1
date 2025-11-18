# PowerShell script to automatically change commit messages
# This uses git filter-branch with a custom message filter

Write-Host "Starting automatic commit message replacement..." -ForegroundColor Green
Write-Host ""

# Backup current branch
$backupBranch = "backup-before-rebase-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
git branch $backupBranch 2>&1 | Out-Null
Write-Host "Created backup branch: $backupBranch" -ForegroundColor Yellow
Write-Host ""

# Set environment variable to suppress warnings
$env:FILTER_BRANCH_SQUELCH_WARNING = "1"

# Get current directory
$currentDir = (Get-Location).Path
$tempFilterPath = Join-Path $currentDir "temp_filter.sh"

# Commit messages mapping
$commits = @(
    @{
        Hash = "55c0e0d298dbb9cb20cb534bcc1014ae0584c71c"
        Short = "55c0e0d"
        Message = "feat: Add RL system management and autonomous mode`n`n- Implement RL control page with system management interface`n- Add RL service with autonomous mode support`n- Create RL API routes for system control`n- Add comprehensive RL API tests`n- Enhance RL service with advanced control features"
    },
    @{
        Hash = "696ad6000c518b4699f22de2624cea803451675d"
        Short = "696ad60"
        Message = "chore: Improve security configurations and phase zero maintenance infrastructure`n`n- Add security headers and middleware improvements`n- Enhance health check endpoints`n- Update docker-compose configuration`n- Improve Kafka service reliability`n- Update setup guide documentation"
    },
    @{
        Hash = "8a934b16a8fb02582a97458efe43aab8f67ba866"
        Short = "8a934b1"
        Message = "feat: Complete backend-dashboard integration and automated CI`n`n- Enhanced frontend pages (HistoricalData, Maintenance, Predictions)`n- Improved backend services (maintenance, prediction, MLflow)`n- Added comprehensive API tests for maintenance and predictions`n- Updated CI/CD workflows`n- WebSocket service improvements"
    }
)

Write-Host "Using git filter-branch to change commit messages..." -ForegroundColor Cyan
Write-Host ""

# Process each commit
foreach ($commit in $commits) {
    Write-Host "Processing commit: $($commit.Short)" -ForegroundColor Yellow
    
    # Escape the message for bash
    $escapedMsg = $commit.Message -replace "`n", "`n" -replace '"', '\"' -replace '\$', '\$'
    
    # Create bash filter script with full path
    $bashScript = @"
#!/bin/bash
if [ "`$GIT_COMMIT" = "$($commit.Hash)" ]; then
    echo -e "$escapedMsg"
else
    cat
fi
"@
    
    # Write to file with full path
    $bashScript | Out-File -FilePath $tempFilterPath -Encoding UTF8 -NoNewline
    
    # Make executable (if on Unix-like system)
    if (Get-Command chmod -ErrorAction SilentlyContinue) {
        chmod +x $tempFilterPath
    }
    
    # Use git filter-branch with full path
    $filterCmd = "bash `"$tempFilterPath`""
    $result = git filter-branch -f --msg-filter $filterCmd -- --all 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Successfully changed commit $($commit.Short)" -ForegroundColor Green
    } else {
        Write-Host "  Failed to change commit $($commit.Short)" -ForegroundColor Red
        Write-Host $result
    }
    
    # Clean up
    Remove-Item $tempFilterPath -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "Commit message replacement completed!" -ForegroundColor Green
Write-Host ""
Write-Host "To verify, run: git log --oneline | Select-String '55c0e0d|696ad60|8a934b1'" -ForegroundColor Cyan
Write-Host ""
Write-Host "IMPORTANT: If commits are already pushed, you will need:" -ForegroundColor Yellow
Write-Host "  git push --force origin Main" -ForegroundColor Yellow
Write-Host ""
Write-Host "WARNING: Force push will rewrite history!" -ForegroundColor Red
Write-Host "Make sure to coordinate with your team before force pushing." -ForegroundColor Red
