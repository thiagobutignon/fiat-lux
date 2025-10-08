#!/bin/bash
# Clean emojis from LaTeX file

cd /Users/thiagobutignon/dev/chomsky/white-paper

# Create backup
cp agi_pt.tex agi_pt.tex.backup

# Replace emojis
sed -i.bak \
  -e 's/✅/$\\checkmark$/g' \
  -e 's/❌/$\\times$/g' \
  -e 's/⚠️/\\textbf{!}/g' \
  -e 's/🔴/\\textcolor{red}{Critical}/g' \
  -e 's/🟡/Medium/g' \
  agi_pt.tex

echo "✓ Emojis replaced in agi_pt.tex"
echo "Backup saved as agi_pt.tex.backup"
