#!/bin/bash
# Script to change commit messages using git rebase

# Find the oldest commit's parent (280f21b is parent of 696ad60)
PARENT_COMMIT="280f21bda5729cbde9d8deeea803a055e074b750"

# Create rebase todo file
cat > /tmp/rebase_todo.txt << 'EOF'
reword 696ad6000c518b4699f22de2624cea803451675d
reword 55c0e0d298dbb9cb20cb534bcc1014ae0584c71c
pick 8a934b16a8fb02582a97458efe43aab8f67ba866
EOF

# Create commit message files
cat > /tmp/msg_696ad60.txt << 'EOF'
chore: Improve security configurations and phase zero maintenance infrastructure

- Add security headers and middleware improvements
- Enhance health check endpoints
- Update docker-compose configuration
- Improve Kafka service reliability
- Update setup guide documentation
EOF

cat > /tmp/msg_55c0e0d.txt << 'EOF'
feat: Add RL system management and autonomous mode

- Implement RL control page with system management interface
- Add RL service with autonomous mode support
- Create RL API routes for system control
- Add comprehensive RL API tests
- Enhance RL service with advanced control features
EOF

cat > /tmp/msg_8a934b1.txt << 'EOF'
feat: Complete backend-dashboard integration and automated CI

- Enhanced frontend pages (HistoricalData, Maintenance, Predictions)
- Improved backend services (maintenance, prediction, MLflow)
- Added comprehensive API tests for maintenance and predictions
- Updated CI/CD workflows
- WebSocket service improvements
EOF

echo "Rebase script created. Run manually:"
echo "  git rebase -i $PARENT_COMMIT"
echo "Then use the message files in /tmp/"

