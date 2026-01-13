#!/bin/bash

# Lokal kaynak dizini
LOCAL_SOURCE="$HOME/Sandbox/github/PHD/data/pdbbind"

# Final hedef dizin (datamover erişilebilir)
REMOTE_TARGET="/g/data/tevfik/pdbbind"

# Datamover node'unu hedefleyen hostname
DATAMOVER_HOST=""

echo "Starting direct transfer via datamover..."
echo "This will:"
echo "1. Request a datamover session on codon"
echo "2. Transfer directly from your local machine to the datamover node"
echo ""

# Önce datamover node adını al ve rsync server başlat
ssh codon << 'ENDSSH'
srun -t 5:00:00 --mem=5G --partition=datamover bash -c "
  echo 'Datamover node ready: '$(hostname)
  echo 'Listening for rsync on port 8873...'
  echo 'You can now run rsync from your local machine to: $(hostname):$REMOTE_TARGET'
  echo 'Command: rsync -avzh --progress $LOCAL_SOURCE rsync://$(hostname):8873/data'
  # Keep session alive for rsync
  sleep 18000
"
ENDSSH

echo "Transfer completed!"
