#!/bin/bash

# GPU nodlarının özetini tablo halinde gösterir
echo "NodeName    CPU  AllocCPU  RAM(GB)   AllocMem(GB)  FreeMem(GB)   GPU            State"
scontrol show node | grep -E "NodeName=codon-gpu" -A 20 | awk '
/NodeName=/ {node=$1; cpu=ram=alloccpu=allocmem=freemem=gres=state=""}
/NodeName=/ {split($1,a,"="); node=a[2]}
/CPUTot=/ {for(i=1;i<=NF;i++){if($i ~ /CPUTot=/){split($i,a,"="); cpu=a[2]}}
           for(i=1;i<=NF;i++){if($i ~ /CPUAlloc=/){split($i,a,"="); alloccpu=a[2]}}}
/RealMemory=/ {for(i=1;i<=NF;i++){if($i ~ /RealMemory=/){split($i,a,"="); ram=a[2]}}
               for(i=1;i<=NF;i++){if($i ~ /AllocMem=/){split($i,a,"="); allocmem=a[2]}}
               for(i=1;i<=NF;i++){if($i ~ /FreeMem=/){split($i,a,"="); freemem=a[2]}}}
/Gres=/ {for(i=1;i<=NF;i++){if($i ~ /Gres=/){split($i,a,"="); gres=a[2]}}}
/State=/ {for(i=1;i<=NF;i++){if($i ~ /State=/){split($i,a,"="); state=a[2]}}}
/Partitions=/ {
  printf "%s %s %s %.1f %.1f %.1f %s %s\n", node, cpu, alloccpu, ram/1024, allocmem/1024, freemem/1024, gres, state
}
' | column -t
