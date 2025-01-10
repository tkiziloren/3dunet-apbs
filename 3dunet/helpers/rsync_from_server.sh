#!bin/bash
rsync -zavz -e ssh codon-slurm-login://homes/tevfik/PHD/3dunet-apbs/3dunet/output/codon  ../output/
echo "Done"

