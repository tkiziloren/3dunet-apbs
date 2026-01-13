#!/bin/bash

# Lokal kaynak dizini
LOCAL_SOURCE="$HOME/Sandbox/github/PHD/data/pdbbind"

# Codon hedef dizin
REMOTE_TARGET="/g/data/tevfik/pdbbind"

# Lokal makinenin hostname/IP'si (değiştirin!)
LOCAL_HOST=$(hostname -f)

echo "Direct transfer using datamover pull method"
echo "================================================"
echo "Local source: $LOCAL_SOURCE"
echo "Remote target: $REMOTE_TARGET"
echo ""
echo "Note: Make sure SSH from codon to your local machine is possible"
echo "      (You may need to add codon's SSH key to your authorized_keys)"
echo ""

# Datamover üzerinden lokal makineden çek
ssh codon << ENDSSH
srun -t 5:00:00 --mem=5G --partition=datamover bash -c "
  echo 'Datamover node: \$(hostname)'
  echo 'Pulling data from $LOCAL_HOST...'
  rsync -avzh --progress $USER@$LOCAL_HOST:$LOCAL_SOURCE/ $REMOTE_TARGET/
  echo 'Transfer completed on \$(date)'
"
ENDSSH

echo "Done!"
