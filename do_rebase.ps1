# Simple script to do rebase with proper editors

Write-Host "Setting up rebase..." -ForegroundColor Green

# Create a simple bash script for sequence editor
$seqEditor = @'
#!/bin/bash
cp rebase_todo.txt "$1"
'@

$seqEditor | Out-File -FilePath "seq_editor.sh" -Encoding UTF8 -NoNewline

# Create a simple bash script for commit message editor
$msgEditor = @'
#!/bin/bash
COMMIT_FILE="$1"

# Try to read the commit hash from the file
COMMIT_HASH=$(git log -1 --format=%H HEAD 2>/dev/null || echo "")

# Check if we're in a rebase
if [ -f ".git/rebase-merge/message" ]; then
    # We're editing a commit message during rebase
    # Try to find which commit we're editing
    if [ -f ".git/rebase-merge/current-files" ]; then
        # This is a reword operation
        # Read the original commit message to identify the commit
        ORIG_MSG=$(cat "$COMMIT_FILE" | head -1)
        
        # Match based on original message content
        if echo "$ORIG_MSG" | grep -q "ط¨ظ‡ط¨ظˆط¯"; then
            # This is 696ad60
            cat msg_696ad60.txt > "$COMMIT_FILE"
        elif echo "$ORIG_MSG" | grep -q "ط§ظپط²ظˆط¯ظ† ظ…ط¯غŒط±غŒطھ"; then
            # This is 55c0e0d
            cat msg_55c0e0d.txt > "$COMMIT_FILE"
        elif echo "$ORIG_MSG" | grep -q "طھظ…ط§ظ… ط³ط§ط®طھ"; then
            # This is 8a934b1
            cat msg_8a934b1.txt > "$COMMIT_FILE"
        fi
    fi
fi
'@

$msgEditor | Out-File -FilePath "msg_editor.sh" -Encoding UTF8 -NoNewline

Write-Host "Scripts created. Now running rebase..." -ForegroundColor Cyan

# Set environment variables
$env:GIT_SEQUENCE_EDITOR = "bash seq_editor.sh"
$env:GIT_EDITOR = "bash msg_editor.sh"

# Run rebase
git rebase -i 280f21b^

Write-Host "Rebase completed!" -ForegroundColor Green

