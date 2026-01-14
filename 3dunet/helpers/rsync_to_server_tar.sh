#!/bin/bash

# İKI AŞAMALI TRANSFER: Önce tar'ı /tmp'ye gönder, sonra datamover'da aç
# Bu yöntem stdin problemi olmadan çalışır

LOCAL_SOURCE="${1:-$HOME/Sandbox/github/PHD/data/pdbbind}"
REMOTE_TARGET="${2:-/g/data/tevfik/pdbbind}"

echo "Two-stage transfer: tar to /tmp, then extract on datamover..."
echo "Source: $LOCAL_SOURCE"
echo "Target: $REMOTE_TARGET"
echo ""

# Temporary file on remote
REMOTE_TMP="/tmp/transfer_$$.tar.gz"

echo "[1/2] Compressing and uploading to /tmp..."
tar cf - -C "$(dirname "$LOCAL_SOURCE")" "$(basename "$LOCAL_SOURCE")" | \
  pigz -1 | \
  ssh -c aes128-gcm@openssh.com -o Compression=no -o ServerAliveInterval=60 codon \
    "cat > $REMOTE_TMP"

echo "[2/2] Extracting on datamover partition..."
ssh codon bash -l <<REMOTE_CMD
srun -t 1:00:00 --mem=5G --partition=datamover bash -c 'pigz -dc $REMOTE_TMP | tar xf - -C $(dirname "$REMOTE_TARGET")'
REMOTE_CMD

echo "[3/2] Cleaning up..."
ssh codon "rm -f $REMOTE_TMP"

echo ""
echo "Transfer completed!"
