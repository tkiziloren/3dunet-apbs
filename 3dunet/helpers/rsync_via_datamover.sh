#!/bin/bash

# Transfer files using rsync via datamover partition
# This creates a wrapper on the remote side that runs rsync on datamover

LOCAL_SOURCE="${1:-/Users/tevfik/Sandbox/github/PHD/data/pdbbind/refined-set-only-used-in-codon-tests/box161}"
REMOTE_TARGET="${2:-/hps/nobackup/arl/chembl/tevfik/deep-apbs-data/pdbbind/refined-set_filter}"

echo "Transfer via datamover partition..."
echo "Source: $LOCAL_SOURCE"
echo "Target: $REMOTE_TARGET"
echo ""

# Create wrapper script on remote
WRAPPER="/tmp/rsync_wrapper_$$.sh"
ssh -c aes128-gcm@openssh.com codon "cat > $WRAPPER" << 'WRAPPER_EOF'
#!/bin/bash -l
# This script runs on datamover and receives files via rsync

# Submit rsync server to datamover
exec srun --partition=datamover --mem=5G --time=5:00:00 \
  rsync --server -vlogDtpre.iLsfxCIvu --partial --inplace . "$1"
WRAPPER_EOF

# Make it executable
ssh codon "chmod +x $WRAPPER"

# Now rsync using this wrapper
rsync -avP \
  --partial \
  --inplace \
  -e "ssh -c aes128-gcm@openssh.com -o Compression=no" \
  --rsync-path="$WRAPPER" \
  "$LOCAL_SOURCE/" \
  codon:"$REMOTE_TARGET"

# Cleanup
ssh codon "rm -f $WRAPPER"

echo ""
echo "Transfer completed!"
