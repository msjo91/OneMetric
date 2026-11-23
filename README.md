# OneMetric

Code for OneMetric, a method that merges LoRA clients at a server in one round. Each client uploads its adapter, the Grams of its layer inputs, and 200 unlabeled calibration inputs. The server builds a set of candidate merges and returns the one with the highest agreement with the clients' own answers on those inputs. No labels, no training, and no second exchange are used.

This repository holds the method, the baselines it is compared with, the preparation of the pools, and the supplement's experiments. It holds no results and no trained adapters.

## Install

```bash
pip install -r requirements.txt  # the versions this code was checked with; the last block of the file is for the LLaVA pools only
```

There is nothing else to install. Run every command below from the repository folder.

The experiments ran on Python 3.13 with PyTorch 2.13 (CUDA 12.6) and Transformers 4.57, one NVIDIA RTX 3090 per run.

Everything is read from and written under one root folder, the repository folder by default. Set `ONEMETRIC_ROOT` to use another one. Results go to `results/`, third-party inputs to `external/`.

## Quick start: one pool

A pool is a set of clients on one backbone. With the inputs in place (next section), this runs the method on the eight-client CLIP ViT-B/32 pool `r8_10` and prints what the server returns:

```bash
scripts/run_pool.sh dc B32 r8_10          # kind, backbone, pool, optional GPU index
python -m onemetric.run_baselines         # with ARCH=B32 CLIPK_SUB=r8_10 in the environment: every baseline row
python -m onemetric.summarize --kind dc --arch B32 --pools r8_10
```

`run_pool.sh` makes six calls: the Chain of Merges merge of the pool, then the method once for each of its four Gram ends (one-shot, sequential, half-sequential, Chain of Merges) and once for the anchored points. `summarize` applies the server's choice across the five run records and compares the returned point with the best baseline row.

## What is where

All modules are in `onemetric/` and run as `python -m onemetric.<name>`. They are configured through environment variables, listed at the top of each file.

| Part | Modules |
|---|---|
| The method | `run_method` (one Gram end per call), `spectral_end` (the spectral end), `summarize` (the server's choice across the runs) |
| Baselines | `run_baselines` (twelve one-shot merges, RegMean, RegMean++, LiNeS, rescaled CT-Merging), `run_chain_of_merges`, `merges`, `merges_ext` |
| Backbones and pools | `clip_backend`, `harness_clip`, `harness_knots`, `siglip_backend`, `harness_siglip`, `dino_backend`, `harness_dino`, `dino_data` |
| Pool preparation | `prepare_pool`, `eval_own_clients`, `data_newtasks`, `data_newtasks_siglip`, `make_newtask_templates`, `download_vtab` |
| Client training | `train_clip_client`, `train_siglip_client`, `train_siglip_client_light`, `train_dino_probe`, `train_dino_client`, and the checks `check_siglip_clients`, `check_dino_probes`, `check_dino_clients` |
| One-shot federated methods | `run_fedfisher_fedlpa`, `run_ghofl_ensemble`, `run_corespace_pgsb` |
| Uploading less | `data_public_clip`, `data_public_siglip_dino`, `data_public_newtasks`, `data_public_draws`, `data_calibration_draws`, `data_calibration_sizes`, `data_gram_sizes`, `gram_noise`, `gram_dp`, `audit_gram_noise`, `audit_gram_size`, `audit_gram_dp`, `audit_adapter` |
| LLaVA pools | `llava_pool`, `llava_ends`, `llava_length`, `llava_sequential_anchor`, `llava_robustmerge`, `llava_ct_pact_ace`, `llava_eval_extra`, and `mm_pool`, `mm_merges`, `mm_eval` |

## Inputs

Models and datasets come from the Hugging Face hub and are cached there. The runs themselves can be offline (`HF_HUB_OFFLINE=1`) once the cache is filled.

| Input | Source | Where it goes |
|---|---|---|
| CLIP ViT-B/32, B/16, L/14 | `openai/clip-vit-base-patch32`, `-base-patch16`, `-large-patch14` | hub cache |
| SigLIP 2 B/16, L/16, So400m | `google/siglip2-base-patch16-224`, `-large-patch16-256`, `-so400m-patch14-224` | hub cache |
| DINOv2 B/14, L/14 | `facebook/dinov2-base`, `facebook/dinov2-large` | hub cache |
| Public CLIP clients, 16 tasks | the rank-16 LoRA adapters released with DC-Merge (Google Drive folders linked from its repository) | `external/dcmerge_ckpt/lora_adapters/<ViT-B-32, ViT-B-16, ViT-L-14>/<task>/` |
| Public CLIP clients, 8 tasks | `hoffman-lab/KnOTS-<backbone>_lora_R16_<task>` | hub cache |
| Benchmark datasets | `tanganke/*`, `ylecun/mnist`, `ufldl-stanford/svhn`, `dpdl-benchmark/oxford_flowers102`, `timm/oxford-iiit-pet`, `zalando-datasets/fashion_mnist`, `clip-benchmark/wds_*` | hub cache |
| Six VTAB tasks | `clip-benchmark/wds_vtab-*`, saved once by `python -m onemetric.download_vtab` | `results/dino2/data/` |
| New tasks | `dpdl-benchmark/caltech101`, `Maysee/tiny-imagenet`, and the FGVC Aircraft archive `fgvc-aircraft-2013b.tar.gz` from its authors' page | hub cache; the archive under `results/clipnew/raw/aircraft/` |
| LLaVA clients and benchmark | LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`), the MM-MergeBench adapters and data (`AuroraZengfh/LLaVA_7B_lora_r16_*`, `AuroraZengfh/MM-MergeBench`), and the benchmark's image sets | `external/mm_mergebench/{adapters, data, images}` |

`external/clip_benchmark/` is part of this repository. It holds the zero-shot class names and prompt templates: those of CLIP Benchmark, the templates of DC-Merge for its eight further tasks, and the templates we wrote for the eight new tasks.

## Preparing the pools

The clients that are not public are trained here, with the recipes of the paper's supplement.

```bash
# 1. the test and calibration draws of a family's tasks (once per family; CLIP shares one cache over its three backbones)
ARCH=B32 CLIPK_SUB=all16 python -m onemetric.harness_clip data
SIG_ARCH=S2B16 CLIPK_SUB=all16 python -m onemetric.harness_siglip data
DINO_ARCH=D2B CLIPK_SUB=all16 python -m onemetric.harness_dino data

# 2. the clients that are not public
SIG_ARCH=S2B16 python -m onemetric.train_siglip_client --task stanford_cars       # SigLIP 2, each of the 16 tasks
DINO_ARCH=D2B python -m onemetric.train_dino_probe --task stanford_cars           # DINOv2: a linear probe on the frozen features,
DINO_ARCH=D2B python -m onemetric.train_dino_client --task stanford_cars          # then the client under that probe
python -m onemetric.data_newtasks caltech101 aircraft tinyimagenet pcam clevr_count dsprites_orientation smallnorb_azimuth dmlab
ARCH=B32 POOL=dc16 python -m onemetric.train_clip_client --task caltech101 --steps 1000 --lr 1e-4 --bs 128 --accum 4 --tag light   # CLIP, each new task
ARCH=B32 POOL=dc16 CLIPK_SUB=k8 python -m onemetric.eval_own_clients              # the new tasks a pool may draw from

# 3. one pool: the pseudo-answers and own accuracies of its clients, and its norm statistics
ARCH=B32 CLIPK_SUB=r8_10 python -m onemetric.prepare_pool
PP_KIND=siglip SIG_ARCH=S2B16 CLIPK_SUB=r8_7 python -m onemetric.prepare_pool
PP_KIND=dino DINO_ARCH=D2B CLIPK_SUB=r8_1 python -m onemetric.prepare_pool
```

## Pools

A pool name selects the tasks. `r<k>_<seed>` draws k of the benchmark tasks with that seed. `all16` is all sixteen, `k8` the eight KnOTS tasks, `s6` the six VTAB tasks on DINOv2. `nall` is the eight new tasks, `nr<k>_<seed>` k of them, and `mx<k>_<seed>` half new and half benchmark tasks.

| Line | Backbones | Pools |
|---|---|---|
| CLIP benchmark | B32, B16, L14 | `r12_10 r12_11 r12_13 r8_10 r8_11 r8_12 r6_13 r6_14 r6_15` |
| CLIP new tasks | B32, B16, L14 | `nall nr6_1 nr6_2 nr6_3 mx12_1 mx12_2 mx12_3 mx8_1 mx8_2 mx8_3 mx6_1 mx6_2` |
| CLIP and SigLIP 2 fresh | L14, S2L16 | `r12_14 r12_15 r12_16 r8_13 r8_14 r8_15 r6_16 r6_17 r6_18 nr6_4 nr6_5 nr6_6 mx12_4 mx12_5 mx12_6 mx8_4 mx8_5 mx8_6 mx6_3 mx6_4 mx6_5` |
| CLIP canonical | B32, B16, L14; B32, L14 | `all16` with kind `dc`; the KnOTS pool with kind `knots` |
| SigLIP 2 benchmark | S2B16, S2L16, S2SO | `all16 k8 r12_7 r12_8 r12_9 r8_7 r8_8 r8_9 r6_10 r6_11 r6_12` |
| SigLIP 2 new tasks | S2B16, S2L16, S2SO | as CLIP new tasks |
| DINOv2 | D2B, D2L | `all16 s6 r12_1 r12_2 r12_3 r8_1 r8_2 r8_3 r6_1 r6_2 r6_3` |

## The supplement's experiments

These are switches of `run_method` and `run_chain_of_merges`; each writes its own results tree and leaves the method's runs untouched.

| Experiment | How |
|---|---|
| Ablations of the spectral end | `MR_ABL=euclid` or `MR_ABL=ctlevels` |
| Further calibration draws | `python -m onemetric.data_calibration_draws` with `CAL_SEED=1` or `2`, then `MR_CALSEED=1` or `2` |
| Public images at the server | `python -m onemetric.data_public_clip --stage pub`, then `MR_PUB=1` |
| Adapter-only upload | `MR_PUB=1 MR_PUBTAG=pubsg MR_PUBC=<folder of public images>` |
| Noise on the uploaded Grams | `MR_PUB=1 MR_PUBTAG=pubg<m> MR_PUBC=...` for noise of relative size m/1000 |
| Grams over more images, and the membership test | `data_gram_sizes`, `audit_gram_size`, then `MR_PUBTAG=pubgn<N>` |
| Release under differential privacy | `audit_gram_dp` writes and attacks the releases, then `MR_PUBTAG=pubst<name>` merges with one |
| Membership test on the adapter | `audit_adapter` |

A variant applies to the one-shot run as it stands. For the sequential ends and the anchored points add `MR_B3NVAR=1`. For the Chain of Merges end, build the merge on the same inputs first (`COM_VAR=<tag>` and, for another image set, `COM_PUBC=<folder>` with `run_chain_of_merges`) and add `MR_B4NVAR=1` to the method call, or `MR_B4NABL=1` for an ablation.

## LLaVA

```bash
python -m onemetric.llava_pool calib && python -m onemetric.llava_pool build      # calibration prompts, baseline merges
python -m onemetric.llava_pool grams && python -m onemetric.llava_pool regmeanpp  # Grams, the Gram ends
python -m onemetric.llava_pool pseudo                                             # the clients' pseudo-answers
python -m onemetric.llava_ends --stage ends                                       # then family, eval, analysis
python -m onemetric.llava_length --stage mixes                                    # then energy, family, eval, analysis
```

`VLM_SUB=V1` to `V6` selects a client composition. `llava_sequential_anchor` adds the sequential and half-sequential ends and the anchored points.

## Notes on reproduction

- Results can move in the fourth decimal of retention with the number of BLAS threads (`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`). The stored runs are reproduced bit for bit with the thread count they used: eight for the main CLIP and SigLIP 2 runs, and six for Chain of Merges, the Chain of Merges end, DINOv2, and the supplement's variants on public images, on the sequential ends, and on the anchored points. With fewer than eight clients the spectral end's orthonormalization is rank-deficient and the result moves by about .002 with the thread count.
- The best baseline row of a pool is the highest test retention among its rows. In the paper, the benchmark lines add LiNeS with labeled beta on task arithmetic, TIES, and Iso-C to the rows that `run_baselines` computes, and they do not use the dense RegMean row or the rescaled CT-Merging rows.

## Tests

```bash
pip install pytest
python -m pytest tests
```

The tests need no data and no GPU. They check the spectral end on random clients (its rank, its orthonormal left directions, its right directions orthonormal in the input metric, and the block energies) and the server's choice (the highest score, the tie order, and the rule for anchored points).

## Third-party material

- `external/clip_benchmark/` contains class names and templates from CLIP Benchmark (MIT license) and templates derived from the DC-Merge repository (MIT license).
- The KnOTS templates in `clip_backend.py` are those of the KnOTS repository (MIT license).
- Chain of Merges, the twelve one-shot merges, RobustMerge, FedFisher, FedLPA, GH-OFL, Core Space, and PGSB are our re-implementations from the papers. No code of theirs is included.
