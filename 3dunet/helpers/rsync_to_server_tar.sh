#!/bin/bash

# EN BASIT YÖNTEM: tar pipe ile direkt transfer
# Home directory'ye alan gerektirmez!

LOCAL_SOURCE="${1:-$HOME/Sandbox/github/PHD/data/pdbbind}"
REMOTE_TARGET="${2:-/g/data/tevfik/pdbbind}"

echo "Transferring via tar+ssh pipe to datamover..."
echo "Source: $LOCAL_SOURCE"
echo "Target: $REMOTE_TARGET"
echo ""

# Tek komut: tar ile sıkıştır, ssh üzerinden gönder, datamover'da aç
# bash -l ile login shell açarak srun'ı PATH'te bulmasını sağlıyoruz
tar czf - -C "$(dirname "$LOCAL_SOURCE")" "$(basename "$LOCAL_SOURCE")" | \
  ssh codon "bash -l -c 'srun -t 5:00:00 --mem=5G --partition=datamover tar xzvf - -C $(dirname "$REMOTE_TARGET")'"

echo ""
echo "Transfer completed!"
