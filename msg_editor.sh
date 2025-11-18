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
        if echo "$ORIG_MSG" | grep -q "ط·آ¨ط¸â€،ط·آ¨ط¸ث†ط·آ¯"; then
            # This is 696ad60
            cat msg_696ad60.txt > "$COMMIT_FILE"
        elif echo "$ORIG_MSG" | grep -q "ط·آ§ط¸ظ¾ط·آ²ط¸ث†ط·آ¯ط¸â€  ط¸â€¦ط·آ¯ط؛إ’ط·آ±ط؛إ’ط·ع¾"; then
            # This is 55c0e0d
            cat msg_55c0e0d.txt > "$COMMIT_FILE"
        elif echo "$ORIG_MSG" | grep -q "ط·ع¾ط¸â€¦ط·آ§ط¸â€¦ ط·آ³ط·آ§ط·آ®ط·ع¾"; then
            # This is 8a934b1
            cat msg_8a934b1.txt > "$COMMIT_FILE"
        fi
    fi
fi