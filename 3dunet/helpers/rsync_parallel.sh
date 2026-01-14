#!/bin/bash

# Parallel rsync using GNU parallel - 4x daha hızlı!
# Requires: brew install parallel

LOCAL_SOURCE="${1:-/Users/tevfik/Sandbox/github/PHD/data/pdbbind/refined-set-only-used-in-codon-tests/box161}"
REMOTE_TARGET="${2:-/hps/nobackup/arl/chembl/tevfik/deep-apbs-data/pdbbind/box161}"
PARALLEL_JOBS="${3:-4}"  # 4 paralel stream

echo "Parallel rsync transfer (${PARALLEL_JOBS} jobs)..."
echo "Source: $LOCAL_SOURCE"
echo "Target: $REMOTE_TARGET"
echo ""

# Create remote directory
ssh codon "mkdir -p $REMOTE_TARGET"

# Find all files and transfer in parallel
find "$LOCAL_SOURCE" -type f | \
  parallel -j $PARALLEL_JOBS \
    rsync -avP \
      --partial \
      --inplace \
      --relative \
      -e "ssh -c aes128-gcm@openssh.com -o Compression=no" \
      {} "codon:$REMOTE_TARGET/../"

echo ""
echo "Parallel transfer completed!"
