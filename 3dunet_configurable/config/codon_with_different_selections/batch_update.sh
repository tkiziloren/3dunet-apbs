#!/bin/bash
for f in *.yml; do
  # accumulation_steps ekle
  if ! grep -q "accumulation_steps" "$f"; then
    awk '/batch_size: 1/ {print; print "  accumulation_steps: 8"; next} 1' "$f" > "${f}.tmp" && mv "${f}.tmp" "$f"
  fi
  
  # model dropout ekle
  if ! grep -q "^model:" "$f"; then
    printf "\nmodel:\n  dropout: 0.2\n" >> "$f"
  fi
done
echo "✅ Updated $(ls *.yml | wc -l) config files"
