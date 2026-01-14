#!/bin/bash

# Fast rsync with optimized compression
LOCAL_SOURCE="${1:-/Users/tevfik/Sandbox/github/PHD/data/pdbbind/refined-set-only-used-in-codon-tests/box161}"
REMOTE_TARGET="${2:-/hps/nobackup/arl/chembl/tevfik/deep-apbs-data/pdbbind}"

echo "Fast rsync transfer to Codon..."
echo "Source: $LOCAL_SOURCE"
echo "Target: $REMOTE_TARGET"
echo ""

# Optimized rsync options:
# -z: compress (but with -z --compress-level=1 for speed)
# --partial: keep partial transfers (resume capability)
# --inplace: update files in-place (faster)
# --no-compress: skip compression for already compressed files (.h5)
# -e "ssh -c aes128-gcm@openssh.com": fastest SSH cipher

rsync -avP \
  --partial \
  --inplace \
  --no-compress \
  --compress-level=1 \
  -e "ssh -c aes128-gcm@openssh.com -o Compression=no" \
  "$LOCAL_SOURCE" "codon:$REMOTE_TARGET/"

echo ""
echo "Transfer completed!"
