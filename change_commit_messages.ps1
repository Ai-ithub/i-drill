# Script to change commit messages from Persian to English
# This script uses git filter-branch to change commit messages

$env:FILTER_BRANCH_SQUELCH_WARNING = "1"

# Commit messages mapping
$commitMessages = @{
    "55c0e0d298dbb9cb20cb534bcc1014ae0584c71c" = @"
feat: Add RL system management and autonomous mode

- Implement RL control page with system management interface
- Add RL service with autonomous mode support
- Create RL API routes for system control
- Add comprehensive RL API tests
- Enhance RL service with advanced control features
"@
    "696ad6000c518b4699f22de2624cea803451675d" = @"
chore: Improve security configurations and phase zero maintenance infrastructure

- Add security headers and middleware improvements
- Enhance health check endpoints
- Update docker-compose configuration
- Improve Kafka service reliability
- Update setup guide documentation
"@
    "8a934b16a8fb02582a97458efe43aab8f67ba866" = @"
feat: Complete backend-dashboard integration and automated CI

- Enhanced frontend pages (HistoricalData, Maintenance, Predictions)
- Improved backend services (maintenance, prediction, MLflow)
- Added comprehensive API tests for maintenance and predictions
- Updated CI/CD workflows
- WebSocket service improvements
"@
}

Write-Host "Starting to change commit messages..."
Write-Host ""

foreach ($commitHash in $commitMessages.Keys) {
    $newMessage = $commitMessages[$commitHash]
    Write-Host "Changing commit: $($commitHash.Substring(0, 7))"
    
    # Create a temporary file with the new message
    $tempFile = [System.IO.Path]::GetTempFileName()
    $newMessage | Out-File -FilePath $tempFile -Encoding UTF8 -NoNewline
    
    # Use git filter-branch
    $filterScript = @"
if [ "`$GIT_COMMIT" = "$commitHash" ]; then
    cat "$tempFile"
else
    cat
fi
"@
    
    $filterScript | Out-File -FilePath "temp_filter.sh" -Encoding UTF8 -NoNewline
    
    Write-Host "  Running filter-branch for $($commitHash.Substring(0, 7))..."
    git filter-branch -f --msg-filter "bash temp_filter.sh" -- --all 2>&1 | Out-Null
    
    Remove-Item "temp_filter.sh" -ErrorAction SilentlyContinue
    Remove-Item $tempFile -ErrorAction SilentlyContinue
    
    Write-Host "  Done!"
    Write-Host ""
}

Write-Host "All commit messages changed successfully!"
Write-Host ""
Write-Host "Note: If commits are already pushed, you'll need:"
Write-Host "  git push --force origin Main"
Write-Host ""
Write-Host "⚠️  WARNING: Force push will rewrite history. Make sure to coordinate with your team!"

