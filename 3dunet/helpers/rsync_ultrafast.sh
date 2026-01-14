#!/bin/bash

# Ultra-fast transfer using rsync with parallel compression
# Requires: brew install pigz (for local compression testing)

LOCAL_SOURCE="${1:-/Users/tevfik/Sandbox/github/PHD/data/pdbbind/refined-set-only-used-in-codon-tests/box161}"
REMOTE_TARGET="${2:-/hps/nobackup/arl/chembl/tevfik/deep-apbs-data/pdbbind/refined-set_filter}"

echo "Ultra-fast transfer with rsync + datamover..."
echo "Source: $LOCAL_SOURCE"  
echo "Target: $REMOTE_TARGET"
echo ""

# Use rsync with custom remote shell that submits to datamover
# This way rsync server runs on datamover partition, not login node
rsync -avzP --partial --inplace \
  -e "ssh -c aes128-gcm@openssh.com -o Compression=no" \
  --rsync-path='bash -l -c "srun --partition=datamover --mem=5G --time=5:00:00 rsync"' \
  "$LOCAL_SOURCE/" \
  codon:"$REMOTE_TARGET/"

echo ""
echo "Ultra-fast transfer completed!"
