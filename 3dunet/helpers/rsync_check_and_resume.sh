#!/bin/bash

# Check what's missing and only send those files
LOCAL_SOURCE="${1:-/Users/tevfik/Sandbox/github/PHD/data/pdbbind/refined-set-only-used-in-codon-tests/box161}"
REMOTE_TARGET="${2:-/hps/nobackup/arl/chembl/tevfik/deep-apbs-data/pdbbind/refined-set_filter}"

echo "Checking remote files via datamover..."
ssh codon "bash -l -c 'srun -t 0:30:00 --mem=2G --partition=datamover bash -c \"ls $REMOTE_TARGET/box161/ 2>/dev/null | wc -l\"'"

echo ""
echo "Checking local files..."
LOCAL_COUNT=$(ls "$LOCAL_SOURCE" 2>/dev/null | wc -l | tr -d ' ')
echo "Local files: $LOCAL_COUNT"

echo ""
read -p "Continue with full rsync? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting rsync..."
    
    # Create temp directory in home
    TEMP_NAME="rsync_$(basename $LOCAL_SOURCE)_$(date +%s)"
    
    echo "Uploading to login node..."
    rsync -avzhP --partial "$LOCAL_SOURCE" "codon:~/$TEMP_NAME/"
    
    echo "Moving to final location via datamover..."
    ssh codon bash -l << ENDSSH
srun -t 5:00:00 --mem=5G --partition=datamover bash -c "
    rsync -avh ~/$TEMP_NAME/ $REMOTE_TARGET/
    rm -rf ~/$TEMP_NAME
    echo 'Verifying files...'
    ls $REMOTE_TARGET/\$(basename $LOCAL_SOURCE)/ | wc -l
"
ENDSSH
    
    echo "Done!"
fi
