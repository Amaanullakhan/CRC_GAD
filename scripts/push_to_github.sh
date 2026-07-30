#!/usr/bin/env bash
# Push CRC_GAD to https://github.com/Amaanullakhan/CRC_GAD
# Run from Git Bash:
#   cd /c/Users/DSU-CSE514-38/Desktop/AMAAN_CRC/CRC_GAD
#   bash scripts/push_to_github.sh

set -e
cd "$(dirname "$0")/.."

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git not found"
  exit 1
fi

if [ ! -d .git ]; then
  git init
fi

git add .

git commit -m "Rebuild CRC-GAD v1.0-rebuild — reproducible experiments and paper tables" || true

git tag -f v1.0-rebuild

if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin https://github.com/Amaanullakhan/CRC_GAD.git
fi

git branch -M main

echo ""
echo "Pushing to origin (you may be asked for GitHub username + PAT)..."
git push -u origin main
git push origin v1.0-rebuild

echo ""
echo "Done: https://github.com/Amaanullakhan/CRC_GAD/tree/v1.0-rebuild"
