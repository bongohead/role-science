#!/usr/bin/env bash
set -euo pipefail

PRIVATE_REPO="/workspace/deliberative-alignment-jailbreaks"
PUBLIC_REPO="/workspace/role-confusion-public"
REF="${1:-HEAD}"

# Paths are relative to the private repo root.
# These are excluded even if they are committed.
BLOCKLIST=(
  "docs/clone.sh"
  "experiments/.dev"
)

cd "$PRIVATE_REPO"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: PRIVATE_REPO is not a git repository: $PRIVATE_REPO" >&2
  exit 1
fi

if [ ! -d "$PUBLIC_REPO/.git" ]; then
  echo "Error: PUBLIC_REPO is not an initialized git repository: $PUBLIC_REPO" >&2
  exit 1
fi

if ! git rev-parse --verify "$REF" >/dev/null 2>&1; then
  echo "Error: git ref does not exist in private repo: $REF" >&2
  exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Note: private repo has uncommitted tracked changes; only committed files from $REF will be exported."
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "Exporting committed files from $PRIVATE_REPO at $REF..."

git archive --format=tar "$REF" | tar -x -C "$TMPDIR"

echo "Applying public blocklist..."

for path in "${BLOCKLIST[@]}"; do
  rm -rf "$TMPDIR/$path"
done

echo "Replacing public repo working tree..."

# Delete everything in the public repo except its .git directory.
find "$PUBLIC_REPO" -mindepth 1 -maxdepth 1 ! -name ".git" -exec rm -rf {} +

# Copy exported committed files into the public repo.
rsync -a "$TMPDIR"/ "$PUBLIC_REPO"/

echo
echo "Public repo synced. Review changes with:"
echo "  cd $PUBLIC_REPO"
echo "  git status"
echo "  git diff --stat"