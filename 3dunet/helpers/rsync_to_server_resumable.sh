#!/bin/bash

# Resumable transfer using rsync via datamover
# Usage: ./rsync_to_server_resumable.sh [source] [target]

LOCAL_SOURCE="${1:-/Users/tevfik/Sandbox/github/PHD/data/pdbbind/refined-set-only-used-in-codon-tests/box161}"
REMOTE_TARGET="${2:-/hps/nobackup/arl/chembl/tevfik/deep-apbs-data/pdbbind/refined-set_filter}"

echo "Resumable transfer to Codon via datamover..."
echo "Source: $LOCAL_SOURCE"
echo "Target: $REMOTE_TARGET"
echo ""

# İki aşamalı: Önce login node'a, sonra datamover ile taşı
TEMP_DIR="~/rsync_temp_$(date +%s)"

echo "Step 1/3: Checking what needs to be transferred..."
ssh codon "bash -l -c 'srun -t 5:00:00 --mem=5G --partition=datamover bash -c \"mkdir -p $REMOTE_TARGET && ls -la $REMOTE_TARGET | head -20\"'"

echo ""
echo "Step 2/3: Syncing to login node (resumable)..."
rsync -avzhP --partial "$LOCAL_SOURCE" codon:"$TEMP_DIR/"

echo ""
echo "Step 3/3: Moving to final destination via datamover..."
ssh codon "bash -l -c 'srun -t 5:00:00 --mem=5G --partition=datamover bash -c \"rsync -avh --progress $TEMP_DIR/ $REMOTE_TARGET/ && rm -rf $TEMP_DIR\"'"

echo ""
echo "Transfer completed!"
