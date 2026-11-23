#!/bin/bash
# OneMetric on one vision pool: the Chain of Merges merge, then the five runs of the method, then the server's choice.
#   scripts/run_pool.sh <kind> <backbone> <pool> [gpu]
#   kind: dc (CLIP, DC-Merge tasks and new tasks) | knots (CLIP, KnOTS tasks) | siglip | dino
#   backbone: B32 B16 L14 | S2B16 S2L16 S2SO | D2B D2L        pool: for example r8_10 (see README, "Pools")
set -e
kind=$1; arch=$2; pool=$3; export CUDA_VISIBLE_DEVICES=${4:-0}
case $kind in
  dc)     P="MR_KIND=dc ARCH=$arch CLIPK_SUB=$pool" ;;
  knots)  P="MR_KIND=knots ARCH=$arch"; pool=8task ;;
  siglip) P="MR_KIND=siglip SIG_ARCH=$arch CLIPK_SUB=$pool" ;;
  dino)   P="MR_KIND=dino DINO_ARCH=$arch CLIPK_SUB=$pool" ;;
  *) echo "kind must be dc, knots, siglip or dino"; exit 1 ;;
esac
PY=${PYTHON:-python}
env $P $PY -m src.method.run_chain_of_merges                 # Chain of Merges merge of the pool (also a baseline row)
env $P $PY -m src.method.run_method                          # one-shot Gram end
env $P MR_SEQ=1 $PY -m src.method.run_method                 # sequential Gram end
env $P MR_SEQ=1 MR_SEQCUT=0.5 $PY -m src.method.run_method   # half-sequential Gram end
env $P MR_COMEND=1 $PY -m src.method.run_method              # Chain of Merges Gram end
env $P MR_ADDON=1 $PY -m src.method.run_method               # anchored points
$PY -m src.method.summarize --kind $kind --arch $arch --pools $pool
