# Simple script to change commit messages using git rebase
# This script prepares the rebase todo file automatically

Write-Host "Preparing git rebase for commit message changes..." -ForegroundColor Green
Write-Host ""

# Find the parent commit of 280f21b (which is parent of 696ad60)
$parentCommit = "280f21bda5729cbde9d8deeea803a055e074b750"

# Create rebase todo file
$rebaseTodo = @"
reword 696ad6000c518b4699f22de2624cea803451675d
reword 55c0e0d298dbb9cb20cb534bcc1014ae0584c71c
reword 8a934b16a8fb02582a97458efe43aab8f67ba866
"@

# Create message files for each commit
$msg1 = @"
chore: Improve security configurations and phase zero maintenance infrastructure

- Add security headers and middleware improvements
- Enhance health check endpoints
- Update docker-compose configuration
- Improve Kafka service reliability
- Update setup guide documentation
"@

$msg2 = @"
feat: Add RL system management and autonomous mode

- Implement RL control page with system management interface
- Add RL service with autonomous mode support
- Create RL API routes for system control
- Add comprehensive RL API tests
- Enhance RL service with advanced control features
"@

$msg3 = @"
feat: Complete backend-dashboard integration and automated CI

- Enhanced frontend pages (HistoricalData, Maintenance, Predictions)
- Improved backend services (maintenance, prediction, MLflow)
- Added comprehensive API tests for maintenance and predictions
- Updated CI/CD workflows
- WebSocket service improvements
"@

# Save files
$rebaseTodo | Out-File -FilePath "rebase_todo.txt" -Encoding UTF8
$msg1 | Out-File -FilePath "msg_696ad60.txt" -Encoding UTF8
$msg2 | Out-File -FilePath "msg_55c0e0d.txt" -Encoding UTF8
$msg3 | Out-File -FilePath "msg_8a934b1.txt" -Encoding UTF8

Write-Host "Files created:" -ForegroundColor Cyan
Write-Host "  - rebase_todo.txt (rebase instructions)" -ForegroundColor White
Write-Host "  - msg_696ad60.txt (message for 696ad60)" -ForegroundColor White
Write-Host "  - msg_55c0e0d.txt (message for 55c0e0d)" -ForegroundColor White
Write-Host "  - msg_8a934b1.txt (message for 8a934b1)" -ForegroundColor White
Write-Host ""
Write-Host "To complete the rebase:" -ForegroundColor Yellow
Write-Host "  1. Set GIT_SEQUENCE_EDITOR to use rebase_todo.txt" -ForegroundColor White
Write-Host "  2. Set GIT_EDITOR to use the message files" -ForegroundColor White
Write-Host ""
Write-Host "Or manually run:" -ForegroundColor Yellow
Write-Host "  git rebase -i $($parentCommit.Substring(0, 7))^" -ForegroundColor White
Write-Host "  Then copy the content from rebase_todo.txt" -ForegroundColor White
Write-Host "  For each commit, copy the message from the corresponding msg_*.txt file" -ForegroundColor White

