#!/bin/bash
# Script to run tests with coverage report

set -e

echo "🧪 Running tests with coverage..."

# Change to backend directory
cd "$(dirname "$0")/../src/backend" || exit 1

# Run tests with coverage
pytest tests/ \
    -v \
    --cov=. \
    --cov-report=html \
    --cov-report=term-missing \
    --cov-report=xml \
    --cov-branch \
    --cov-fail-under=70

# Check if coverage report was generated
if [ -d "htmlcov" ]; then
    echo ""
    echo "✅ Coverage report generated in htmlcov/index.html"
    echo "📊 Open the report with: open htmlcov/index.html"
else
    echo "⚠️  Coverage report not generated"
    exit 1
fi

