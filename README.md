# Self-Supervised Skeleton-Based Action Recognition

Teacher–student learning from 3D skeleton sequences, with masked feature prediction,
motion reconstruction and downstream action recognition. The code includes parallel
and coupled transformer variants, with optional GradNorm loss balancing.

**Ruihang Jiang · VITA Lab, EPFL · Master's semester project · Mar–Sep 2024**  
Supervisor: Prof. Alexandre Alahi · Advisor: Mohamed Abdelfattah

## Approach
The masked student predicts features supplied by an exponential-moving-average teacher.
A decoder reconstructs motion targets. Fine-tuning and linear probing subsequently
use action labels; pretraining is self-supervised.

## Code map
| Purpose | Parallel | Coupled |
|---|---|---|
| Pretraining | `pretrain_main_parallel.py` | `pretrain_main_couple.py` |
| Fine-tuning | `finetune_main_parallel.py` | `finetune_main_couple.py` |
| Linear probe | `linprobe_main_parallel.py` | `linprobe_main_couple.py` |

The `_GradNorm.py` files contain loss-balancing variants. Models are in `models/`,
sequence loaders in `feeder/`, and experiment settings in `config/`.

## Environment and data
`requirements.txt` is an unpinned dependency inventory. The historical environment
is not locked; legacy timm APIs require compatibility checks before long runs.
Obtain NTU RGB+D from https://rose1.ntu.edu.sg/dataset/actionRecognition/.
Configs expect a preprocessed archive such as `dataset/NTU60/NTU60_XSub_kf.npz`.
Review `feeder/feeder_ntu.py` for array fields and shapes. Raw sequences, checkpoints
and the original subset-selection artifact are not included.

## Entry points
After preparing data, dependencies and checkpoint paths in the YAML configs:

```bash
python pretrain_main_parallel.py --config config/ntu60_xsub_pretrain_parallel.yaml
python finetune_main_parallel.py --config config/ntu60_xsub_finetune_parallel.yaml
python linprobe_main_parallel.py --config config/ntu60_xsub_linear_parallel.yaml
```

These are entry-point examples, not verified end-to-end reproduction commands.
The scripts launch CUDA workers themselves; restrict devices to your job allocation.
For a dataset-free synthetic model test, run `python smoke_test.py`.

## Experimental evidence
The earlier report used **one quarter of NTU RGB+D 60**, with a cross-subject split.
Table 1 reported 82.3% top-1 / 95.0% top-5 for a spatial/patch/decoder model.
That early implementation differs from this later repository version.

A later retained log, `output_dir_finetune_Parallel_5+3`, records 86.25% best top-1
at zero-based epoch 89, and 95.93% top-5 at that epoch. This is a historical log
observation, not an independently reproduced benchmark. Its exact data/config/checkpoint
mapping requires confirmation. Neither figure should be advertised as full-NTU60
benchmark performance. See `docs/RESULT_PROVENANCE.md`.

## Acknowledgements and licensing
Conducted at VITA Lab under Alexandre Alahi and Mohamed Abdelfattah. The project
builds on MAMP, MAE, DeiT, BEiT and upstream transformer utilities. Existing copyright
notices remain in place. This update does not relicense the code; see `THIRD_PARTY.md`.

## Validation
See `VALIDATION.md` for syntax and synthetic model checks. Full training and reproduction
of historical results have not been rerun.
