#!/bin/bash

# Tek seferde çalıştırmak için
# Kullanım: ./rsync_to_server_oneliner.sh /path/to/local/files /g/data/tevfik/target

LOCAL_SOURCE="${1:-$HOME/Sandbox/github/PHD/data/pdbbind}"
REMOTE_TARGET="${2:-/g/data/tevfik/pdbbind}"

# Önce login'e gönder, sonra datamover ile taşı
ssh codon "mkdir -p ~/tmp_transfer" && \
rsync -zavh --progress "$LOCAL_SOURCE" codon:~/tmp_transfer/ && \
ssh codon "srun -t 1:30:30 --mem=5G --partition=datamover bash -c 'rsync -avh --progress ~/tmp_transfer/ $REMOTE_TARGET/ && rm -rf ~/tmp_transfer'"

echo "Done!"
