#!/usr/bin/env bash
# Push local robot_telemetry_lab to https://github.com/khushidonda/robot_telemetry_lab
# Run from your Mac (requires GitHub auth: gh auth login, SSH key, or credential helper).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ORIGIN="${GITHUB_REMOTE:-https://github.com/khushidonda/robot_telemetry_lab.git}"

echo "[push] repo: $REPO_ROOT"
echo "[push] remote: $ORIGIN"

if [ ! -d .git ]; then
  git init
  git branch -M main
fi

# Prefer your local identity if unset (edit here once if needed).
if ! git config user.email >/dev/null 2>&1; then
  git config user.email "khushidonda@users.noreply.github.com"
fi
if ! git config user.name >/dev/null 2>&1; then
  git config user.name "khushidonda"
fi

git remote remove origin 2>/dev/null || true
git remote add origin "$ORIGIN"

git add -A
git status -sb

if git diff --cached --quiet && git diff --quiet; then
  echo "[push] nothing to commit (working tree clean)"
else
  git commit -m "Robot telemetry lab: dashboard, analyzers, detectors, and docs"
fi

git fetch origin main

# Remote may contain only GitHub's README initial commit; merge without losing local files.
if git show-ref --verify --quiet "refs/remotes/origin/main"; then
  if git merge-base HEAD origin/main >/dev/null 2>&1; then
    echo "[push] histories already related; pulling merge if needed"
    git pull origin main --no-edit || true
  else
    echo "[push] unrelated histories; merging origin/main (prefer local on conflicts)"
    git merge origin/main --allow-unrelated-histories -X ours --no-edit
  fi
fi

git push -u origin main
echo "[push] done"
