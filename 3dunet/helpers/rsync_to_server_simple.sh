#!/bin/bash

# Kullanım: ./rsync_to_server_simple.sh [kaynak_dizin] [hedef_dizin]

LOCAL_SOURCE="${1:-$HOME/Sandbox/github/PHD/data/pdbbind}"
REMOTE_TARGET="${2:-/g/data/tevfik/pdbbind}"

echo "Simple datamover transfer"
echo "========================="
echo "Source: $LOCAL_SOURCE"
echo "Target: codon:$REMOTE_TARGET"
echo ""

# Tek komutla: SSH ile datamover session aç ve rsync beklet
ssh codon "srun -t 5:00:00 --mem=5G --partition=datamover bash" << COMMANDS &
# Datamover node'da rsync daemon başlat
mkdir -p $REMOTE_TARGET
cd $REMOTE_TARGET
echo "Ready to receive on \$(hostname) at $REMOTE_TARGET"
# Keep alive
tail -f /dev/null
COMMANDS

# Datamover session hazır olana kadar bekle
sleep 10

# Şimdi pipe üzerinden transfer et
echo "Starting transfer via SSH pipe..."
tar czf - -C "$(dirname "$LOCAL_SOURCE")" "$(basename "$LOCAL_SOURCE")" | \
  ssh codon "srun -t 5:00:00 --mem=5G --partition=datamover tar xzf - -C $(dirname "$REMOTE_TARGET")"

echo "Transfer completed!"
