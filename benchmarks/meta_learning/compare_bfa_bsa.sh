#!/usr/bin/bash
for L in 128 256 512 1024 2048 4096; do
  echo 'BFA vs. BSA:'
  ./test_biased_flex_attention.py -L $L
  ./test_biased_scan_attention.py -L $L
  echo ''
done
