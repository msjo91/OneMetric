# OneMetric

OneMetric merges LoRA clients at a server in one round. Each client uploads its adapter, the Grams of its layer inputs, and 200 unlabeled calibration inputs. The server builds candidate merges and returns the one that agrees most with the clients' own answers on those inputs. It uses no labels, no training, and no second exchange.

This repository holds the code only: the method, the baselines, pool preparation, and the supplement's experiments.

## Setup

```bash
pip install -r requirements.txt
```

Run every command from the repository folder. Results are written to `results/` and third-party inputs are read from `external/`. Set `ONEMETRIC_ROOT` to keep both elsewhere. The experiments ran on Python 3.13 with one NVIDIA RTX 3090 per run.

## Layout

| Folder | Content |
|---|---|
| `src/method` | the method: `run_method`, `spectral_end`, `run_chain_of_merges`, `summarize`, `prepare_pool` |
| `src/backbones` | CLIP, SigLIP 2, and DINOv2: models, tasks, and data draws |
| `src/baselines` | the one-shot merges (`run_baselines`) and the one-shot federated methods |
| `src/clients` | client training, checks, and the data of the new tasks |
| `src/supplement` | public images, calibration draws, Gram noise, the formal guarantee, and the audits |
| `src/llava` | the LLaVA-1.5 pools |

A module runs as `python -m src.<folder>.<name>`. Its settings are environment variables, listed at the top of its file.

## Run one pool

A pool is a set of clients on one backbone. This runs the method and the baselines on the eight-client CLIP ViT-B/32 pool `r8_10`, then prints what the server returns:

```bash
scripts/run_pool.sh dc B32 r8_10                                  # kind, backbone, pool, optional GPU index
ARCH=B32 CLIPK_SUB=r8_10 python -m src.baselines.run_baselines
python -m src.method.summarize --kind dc --arch B32 --pools r8_10
```

Kinds and backbones: `dc` or `knots` with `B32 B16 L14`, `siglip` with `S2B16 S2L16 S2SO`, `dino` with `D2B D2L`.

Pool names: `r<k>_<seed>` draws k benchmark tasks, `all16` is all sixteen, `k8` the eight KnOTS tasks, `s6` the six VTAB tasks on DINOv2, `nall` the eight new tasks, `nr<k>_<seed>` k of them, and `mx<k>_<seed>` half new and half benchmark tasks.

## Prepare the inputs

Models and datasets come from the Hugging Face hub. Three inputs are fetched by hand:

- the DC-Merge LoRA adapters, from the links in its repository, into `external/dcmerge_ckpt/lora_adapters/<ViT-B-32, ViT-B-16, ViT-L-14>/<task>/`
- the FGVC Aircraft archive `fgvc-aircraft-2013b.tar.gz`, into `results/clipnew/raw/aircraft/`
- for LLaVA, the MM-MergeBench adapters, data, and images, into `external/mm_mergebench/{adapters, data, images}`

Then, per family:

```bash
# 1. test and calibration draws of the tasks
ARCH=B32 CLIPK_SUB=all16 python -m src.backbones.harness_clip data
SIG_ARCH=S2B16 CLIPK_SUB=all16 python -m src.backbones.harness_siglip data
python -m src.clients.download_vtab && DINO_ARCH=D2B CLIPK_SUB=all16 python -m src.backbones.harness_dino data

# 2. the clients that are not public (one call per task)
SIG_ARCH=S2B16 python -m src.clients.train_siglip_client --task stanford_cars
DINO_ARCH=D2B python -m src.clients.train_dino_probe --task stanford_cars
DINO_ARCH=D2B python -m src.clients.train_dino_client --task stanford_cars
python -m src.clients.data_newtasks caltech101 aircraft tinyimagenet pcam clevr_count dsprites_orientation smallnorb_azimuth dmlab
ARCH=B32 POOL=dc16 python -m src.clients.train_clip_client --task caltech101 --steps 1000 --lr 1e-4 --bs 128 --accum 4 --tag light

# 3. one pool
ARCH=B32 CLIPK_SUB=r8_10 python -m src.method.prepare_pool        # PP_KIND=siglip or dino for the other families
```

## Supplement experiments

Each is a switch of `run_method` and writes its own results tree.

| Experiment | Switch |
|---|---|
| Ablations of the spectral end | `MR_ABL=euclid` or `MR_ABL=ctlevels` |
| Further calibration draws | `src.supplement.data_calibration_draws` with `CAL_SEED=1`, then `MR_CALSEED=1` |
| Public images at the server | `src.supplement.data_public_clip --stage pub`, then `MR_PUB=1` |
| Adapter-only upload | `MR_PUB=1 MR_PUBTAG=pubsg MR_PUBC=<public image folder>` |
| Noise on the Grams | `MR_PUB=1 MR_PUBTAG=pubg<m> MR_PUBC=...`, noise of relative size m/1000 |
| Larger Grams and membership test | `data_gram_sizes`, `audit_gram_size`, then `MR_PUBTAG=pubgn<N>` |
| Formal guarantee | `audit_gram_dp`, then `MR_PUBTAG=pubst<name>` |
| Membership test on the adapter | `audit_adapter` |

On the sequential ends and anchored points add `MR_B3NVAR=1`. On the Chain of Merges end add `MR_B4NVAR=1`, or `MR_B4NABL=1` for an ablation.

## LLaVA

```bash
python -m src.llava.llava_pool calib && python -m src.llava.llava_pool build
python -m src.llava.llava_pool grams && python -m src.llava.llava_pool regmeanpp && python -m src.llava.llava_pool pseudo
python -m src.llava.llava_ends --stage ends
python -m src.llava.llava_length --stage mixes
```

`VLM_SUB=V1` to `V6` selects a client composition.

## Notes

- Retention can move in the fourth decimal with the number of BLAS threads (`OPENBLAS_NUM_THREADS`). The paper's runs used eight for the main CLIP and SigLIP 2 runs, and six for Chain of Merges, DINOv2, and the supplement's variants.
- The paper's best baseline row on the benchmark pools differs slightly from the rows of `run_baselines`: it adds LiNeS with a labeled scale on task arithmetic, TIES, and Iso-C, and leaves out the dense RegMean row and the rescaled CT-Merging rows.
- Tests: `pip install pytest && python -m pytest tests`. They need no data and no GPU.

## Third-party material

`external/clip_benchmark/` holds class names and prompt templates from CLIP Benchmark and DC-Merge (both MIT), plus the templates we wrote for the new tasks. The KnOTS templates in `src/backbones/clip_backend.py` are from the KnOTS repository (MIT). All baselines are our re-implementations from the papers.
