#!/bin/bash

# AGI Recursive System - Run All Demos
# Executes all 4 demos in sequence with cost tracking

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  AGI RECURSIVE SYSTEM - Complete Demo Suite                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "This will run all 4 demos in sequence:"
echo "  1. Anthropic Adapter Demo (~$0.007)"
echo "  2. Slice Navigation Demo ($0)"
echo "  3. Anti-Corruption Layer Demo ($0)"
echo "  4. Budget Homeostasis Demo (~$0.02-0.05)"
echo ""
echo "Total estimated cost: ~$0.03-0.06 (~R$0.15-0.30)"
echo ""
read -p "Press ENTER to continue or Ctrl+C to cancel..."
echo ""

# Track start time
START_TIME=$(date +%s)

# Demo 1: Anthropic Adapter
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🤖 DEMO 1/4: Anthropic Adapter"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
npx tsx src/agi-recursive/examples/anthropic-adapter-demo.ts
echo ""
echo "✅ Demo 1 completed"
echo ""
read -p "Press ENTER to continue to Demo 2..."
echo ""

# Demo 2: Slice Navigation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧭 DEMO 2/4: Slice Navigation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
npx tsx src/agi-recursive/examples/slice-navigation-demo.ts
echo ""
echo "✅ Demo 2 completed"
echo ""
read -p "Press ENTER to continue to Demo 3..."
echo ""

# Demo 3: ACL Protection
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🛡️  DEMO 3/4: Anti-Corruption Layer"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
npx tsx src/agi-recursive/examples/acl-protection-demo.ts
echo ""
echo "✅ Demo 3 completed"
echo ""
read -p "Press ENTER to continue to Demo 4 (FINAL - Full AGI System)..."
echo ""

# Demo 4: Budget Homeostasis (Full System)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧠 DEMO 4/4: Budget Homeostasis (FULL AGI SYSTEM)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
npx tsx src/agi-recursive/examples/budget-homeostasis.ts
echo ""
echo "✅ Demo 4 completed"
echo ""

# Calculate total time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

# Summary
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  ALL DEMOS COMPLETED SUCCESSFULLY! 🎉                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Summary:"
echo "  ✅ Demos completed: 4/4"
echo "  ⏱️  Total time: ${MINUTES}m ${SECONDS}s"
echo "  💰 Estimated cost: ~\$0.03-0.06 (~R\$0.15-0.30)"
echo ""
echo "📚 Next Steps:"
echo "  - Review outputs above"
echo "  - Read docs/AGI_QUICKSTART.md for details"
echo "  - Try modifying queries in the demos"
echo "  - Create your own specialized agents"
echo ""
echo "🔗 Resources:"
echo "  - PR #11: https://github.com/thiagobutignon/fiat-lux/pull/11"
echo "  - README: See 'AGI Recursive System' section"
echo "  - CHANGELOG.md: Detailed feature documentation"
echo ""
