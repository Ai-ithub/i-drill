#!/bin/bash
# Script to run weekly security scan locally
# This script mimics what the GitHub Actions workflow does

set -e  # Exit on error

echo "🔒 Starting Weekly Security Scan..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "requirements/dev.txt" ]; then
    echo -e "${RED}❌ Error: requirements/dev.txt not found${NC}"
    echo "Please run this script from the i-drill directory"
    exit 1
fi

# Create reports directory
REPORTS_DIR="security-reports-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$REPORTS_DIR"
echo "📁 Reports will be saved to: $REPORTS_DIR"
echo ""

# Install security tools if not already installed
echo "📦 Installing security tools..."
pip install -q bandit[toml] pip-audit safety || {
    echo -e "${RED}❌ Failed to install security tools${NC}"
    exit 1
}
echo -e "${GREEN}✅ Security tools installed${NC}"
echo ""

# ==================== Bandit Scan ====================
echo "🔍 Running Bandit scan..."
if bandit -r src/ -f json -o "$REPORTS_DIR/bandit-report.json" -ll --exclude src/backend/tests,src/tests 2>/dev/null; then
    echo -e "${GREEN}✅ Bandit scan completed${NC}"
    bandit -r src/ -f screen -ll --exclude src/backend/tests,src/tests 2>&1 | tee "$REPORTS_DIR/bandit-console.txt"
else
    echo -e "${YELLOW}⚠️  Bandit found issues (check report)${NC}"
    bandit -r src/ -f screen -ll --exclude src/backend/tests,src/tests 2>&1 | tee "$REPORTS_DIR/bandit-console.txt"
fi
echo ""

# ==================== pip-audit ====================
echo "🔍 Running pip-audit..."
for req_file in requirements/backend.txt requirements/ml.txt requirements/dev.txt; do
    if [ -f "$req_file" ]; then
        echo "  Checking $req_file..."
        pip-audit --requirement "$req_file" --format json --output "$REPORTS_DIR/pip-audit-$(basename $req_file).json" --desc 2>/dev/null || true
        pip-audit --requirement "$req_file" --desc 2>&1 | tee "$REPORTS_DIR/pip-audit-$(basename $req_file)-console.txt"
    fi
done
echo -e "${GREEN}✅ pip-audit completed${NC}"
echo ""

# ==================== Safety Check ====================
echo "🔍 Running Safety check..."
if safety check --file requirements/backend.txt --file requirements/ml.txt --file requirements/dev.txt --json --output "$REPORTS_DIR/safety-report.json" 2>/dev/null; then
    echo -e "${GREEN}✅ Safety check passed${NC}"
else
    echo -e "${YELLOW}⚠️  Safety found issues (check report)${NC}"
    safety check --file requirements/backend.txt --file requirements/ml.txt --file requirements/dev.txt 2>&1 | tee "$REPORTS_DIR/safety-console.txt"
fi
echo ""

# ==================== Generate Summary ====================
echo "📊 Generating summary..."
cat > "$REPORTS_DIR/SUMMARY.md" << EOF
# Security Scan Summary

**Date**: $(date)
**Reports Directory**: $REPORTS_DIR

## Reports Generated

- ✅ Bandit Report: \`bandit-report.json\`
- ✅ pip-audit Reports: \`pip-audit-*.json\`
- ✅ Safety Report: \`safety-report.json\`

## View Reports

\`\`\`bash
# View JSON reports
cat $REPORTS_DIR/*.json

# View console outputs
cat $REPORTS_DIR/*-console.txt

# View this summary
cat $REPORTS_DIR/SUMMARY.md
\`\`\`

## Next Steps

1. Review all reports
2. Fix Critical and High severity issues
3. Update dependencies if needed
4. Re-run scan after fixes

EOF

echo -e "${GREEN}✅ Summary generated: $REPORTS_DIR/SUMMARY.md${NC}"
echo ""

# ==================== Final Summary ====================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ Weekly Security Scan Completed!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📁 Reports saved to: $REPORTS_DIR"
echo "📄 Summary: $REPORTS_DIR/SUMMARY.md"
echo ""
echo "Next steps:"
echo "  1. Review reports in $REPORTS_DIR"
echo "  2. Fix Critical/High severity issues"
echo "  3. Update dependencies if needed"
echo ""




