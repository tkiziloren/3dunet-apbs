#!/bin/bash

# Lokal kaynak dizini (değiştirin)
LOCAL_SOURCE="$HOME/Sandbox/github/PHD/data/pdbbind"

# Codon'daki hedef dizin (değiştirin)
REMOTE_TARGET="/g/data/tevfik/pdbbind"

# Önce SSH üzerinden datamover session başlat ve rsync çalıştır
ssh codon << 'ENDSSH'
srun -t 1:30:30 --mem=5G --partition=datamover bash -c "
  # rsync'i datamover node'undan çalıştır
  rsync -zavh --progress $USER@codon-slurm-login:$LOCAL_SOURCE $REMOTE_TARGET
"
ENDSSH

echo "Transfer completed!"
