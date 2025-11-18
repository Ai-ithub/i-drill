#!/bin/bash
# Auto editor for git commit message - replaces commit message based on commit hash

COMMIT_HASH="$1"

case "$COMMIT_HASH" in
    696ad6000c518b4699f22de2624cea803451675d)
        cat msg_696ad60.txt
        ;;
    55c0e0d298dbb9cb20cb534bcc1014ae0584c71c)
        cat msg_55c0e0d.txt
        ;;
    8a934b16a8fb02582a97458efe43aab8f67ba866)
        cat msg_8a934b1.txt
        ;;
    *)
        # For other commits, use the original message
        cat
        ;;
esac

