# Reversible Data Hiding: Triple-Hybrid Predictor Framework

## Project Overview
This research project implements an extended reversible data hiding (RDH) framework based on:
> **Baseline Paper:** *“CNN-Based Hybrid Prediction Scheme for Image Reversible Data Hiding”* by Yi Zhao et al., published online in 2025 and appearing in *Circuits, Systems, and Signal Processing* (2026), DOI: [10.1007/s00034-025-03332-1](https://doi.org/10.1007/s00034-025-03332-1).

### Core Research Extension
The baseline paper uses a dual hybrid predictor: Rhombus Predictor (RP) for smooth pixels and a Convolutional Neural Network (CNNP) for complex pixels. This research extends the framework to a **Triple-Hybrid Predictor**:
1. **Rhombus Predictor (RP):** Fast, rule-based 4-neighbor averaging for low-complexity / smooth regions.
2. **CNN Predictor (CNNP):** Receptive field feature extraction for medium-complexity regions.
3. **Transformer Predictor (TFP):** Lightweight, window-restricted self-attention predictor for high-complexity / directional texture regions.
4. **Complexity Router:** Evaluates local context variance / directional gradient energy and assigns pixels dynamically using two tunable thresholds $(T_1, T_2)$.

---

## Workspace & Project Structure
The repository is modularly organized under `D:\rdh-triple-hybrid\`:

```
d:\rdh-triple-hybrid\
├── .venv\                    # Isolated Python 3.12.10 virtual environment
├── configs\                  # Configuration files
├── src\
│   ├── __init__.py
│   ├── config.py             # Central dataclass configuration (paths, seeds, hardware, thresholds)
│   ├── data\                 # Image loaders, checkerboard partitioning, preprocessing
│   │   └── __init__.py
│   ├── predictors\           # RP, CNNP, lightweight Transformer, and Complexity Router
│   │   └── __init__.py
│   └── rdh\                  # PEE embedding/decoding, overflow/underflow map handling
│       └── __init__.py
├── scripts\
│   ├── create_splits.py      # Reproducible split generator (seed=42) and manifest builder
│   └── check_leakage.py      # Automated SHA-256 content and filename zero-leakage verifier
├── tests\
│   ├── __init__.py
│   └── test_data_config.py   # Unit tests for config loading, splits, and leakage prevention
├── data\
│   ├── raw\                  # Read-only references to raw image directories
│   └── processed\            # Generated splits.json and dataset_manifest.json
├── outputs\
│   ├── figures\              # EDA plots, training curves, prediction error distributions
│   └── tables\               # Benchmark evaluation metrics (MAE, RMSE, P(e=0), runtime)
├── checkpoints\              # Trained PyTorch model weights (.pt)
├── paper\                    # IEEE LaTeX draft, bibliography, and figures
├── requirements.txt          # Python dependencies
└── README.md                 # Setup and documentation guide
```

---

## Python Virtual Environment & Setup

### Environment Activation
In Windows PowerShell:
```powershell
# Navigate to project directory
cd D:\rdh-triple-hybrid

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### Proposed Package Installation Command
> **Disk Protection Note:** Drive `C:` has ~7.95 GB free space, whereas Drive `D:` has 272 GB free space. To avoid exhausting `C:` drive with pip wheel caches, all pip caching is routed to `D:\.pip_cache`.

```powershell
# Step 1: Install core PyTorch with CUDA 12.4 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --cache-dir D:\.pip_cache

# Step 2: Install data, scientific, and testing dependencies
pip install numpy pillow scipy matplotlib pandas pytest --cache-dir D:\.pip_cache
```

### Hardware & CUDA Compatibility Matrix
- **GPU:** NVIDIA GeForce RTX 2050 Laptop GPU (4,096 MiB VRAM)
- **Architecture:** Ampere (GA107, Compute Capability `sm_86`)
- **NVIDIA Driver:** 552.27 (Supports CUDA up to 12.4)
- **PyTorch Build:** `torch==2.6.0+cu124` (Includes native PTX/cubin kernels for `sm_86`)
- **Python Compatibility:** Python 3.12.10 (Fully supported by PyTorch 2.6.0)

---

## Dataset Splits & Integrity Status

The dataset splits are deterministically defined in `data/processed/splits.json` and recorded in `data/processed/dataset_manifest.json`:

| Split | Source Dataset | Image Count | Resolution | Format / Channels | Role |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Train** | BOSSbase 1.01 (`C:\Users\mails\Downloads\BOSSbase_1.01`) | **3,500** | $512 \times 512$ | PGM / 1-channel mono | CNN & Transformer training only |
| **Validation** | BOSSbase 1.01 | **500** | $512 \times 512$ | PGM / 1-channel mono | Loss monitoring & $(T_1, T_2)$ threshold tuning |
| **Holdout** | BOSSbase 1.01 | **6,000** | $512 \times 512$ | PGM / 1-channel mono | Locked; unused unless explicitly approved |
| **Test** | USC-SIPI Misc (`C:\Users\mails\Downloads\misc\misc`) | **19** | $512 \times 512$ | TIFF / RGB & mono | Held-out benchmark evaluation |

### Automated Leakage Check Execution
Run the automated verification script:
```powershell
python scripts/check_leakage.py
```
The audit recomputes SHA-256 directly from every file in Train, Validation,
Holdout, and Test. It detects duplicate content within a split as well as
across splits. The current split lists remain unchanged, but the audit reports
these known exact-content matches:

- Train `5538.pgm` ↔ Holdout `5343.pgm`
- Train `5539.pgm` ↔ Holdout `5344.pgm`
- Train `5342.pgm` ↔ Holdout `5537.pgm`
- Train `5341.pgm` ↔ Train `5536.pgm` (duplicate within Train)

The historical split is **not content-independent**; results produced from
historical checkpoints must not be described as an independent BOSSbase
holdout evaluation. The versioned v2 split below resolves cross-partition
duplicates without replacing historical splits or results.
Create and audit the clean versioned split with:

```powershell
.\.venv\Scripts\python.exe scripts\create_clean_split_v2.py
.\.venv\Scripts\python.exe scripts\check_leakage.py --versioned v2
```

Version 2 is stored only in `data/processed/v2/`: three redundant holdout
copies are excluded and the two duplicate Train filenames remain together in
Train. Active counts are Train 3,500, Validation 500, Holdout 5,997, and Test
19. A fresh four-partition SHA-256 audit passed with no cross-partition content
overlap. No historical split or result is changed.

### Clean-v2 Training and Evaluation

The clean-v2 CNN and Transformer checkpoints were trained using only the
3,500-image v2 training split. Model selection and router tuning used only the
500-image v2 validation split. Frozen checkpoints and thresholds were then
evaluated on the 5,997-image v2 holdout. The holdout was not used for training,
checkpoint selection, or threshold tuning. This clean-v2 evaluation is
separate from the historical experiment.

Reproduce the workflow with these commands (each output path is exclusive and
must not already exist):

```powershell
.\.venv\Scripts\python.exe scripts\train_cnn.py --dataset-version v2 --checkpoint-dir checkpoints\clean_v2 --output-dir outputs\clean_v2_experiment\cnn_training --seed 42
.\.venv\Scripts\python.exe scripts\train_transformer.py --dataset-version v2 --checkpoint-dir checkpoints\clean_v2 --output-dir outputs\clean_v2_experiment\transformer_training --seed 42
.\.venv\Scripts\python.exe scripts\tune_router.py --dataset-version v2 --checkpoint-dir checkpoints\clean_v2 --output-dir outputs\clean_v2_experiment\router_tuning
.\.venv\Scripts\python.exe scripts\evaluate_clean_v2_holdout.py --checkpoint-dir checkpoints\clean_v2 --thresholds outputs\clean_v2_experiment\router_tuning\router_thresholds.json --output-dir outputs\clean_v2_experiment\holdout_evaluation
.\.venv\Scripts\python.exe scripts\evaluate_rdh_benchmark_v1.py --dataset-version v2 --checkpoint-dir checkpoints\clean_v2 --thresholds outputs\clean_v2_experiment\router_tuning\router_thresholds.json --output-dir outputs\clean_v2_experiment\usc_sipi_rdh_v1
```

If holdout evaluation is interrupted, `--resume` may be used only with the
same split, validation-selected thresholds, and checkpoint fingerprints; it
validates the existing manifest and complete per-image rows before appending.

Verified clean-v2 training durations were 363.4 seconds (CNN) and 1,205.4
seconds (Transformer). Best validation masked-L1 MAE was 3.0396 for CNN and
3.0207 for Transformer. Validation-only threshold tuning selected `T1=1.0`,
`T2=inf` with mean validation predictor MAE 2.5557. The existing router
represents infinite `T2` with its finite `1e9` dual-hybrid sentinel, so this
selected configuration routes no pixels to the Transformer.

On the v2 BOSSbase holdout, predictor MAE in grayscale levels (mean ± sample
standard deviation across 5,997 images) was:

| Configuration | Predictor MAE |
| :--- | ---: |
| RP-only | 3.3299 ± 2.3491 |
| CNN-only | 2.5986 ± 2.0445 |
| Transformer-only | 2.6137 ± 2.0382 |
| Triple-hybrid router | 2.5943 ± 2.0447 |

These are predictor metrics, not RDH distortion or payload-capacity results.
The separate 19-image USC-SIPI RDH benchmark completed 76/76 configurations
with exact payload and grayscale-cover recovery. Each image used a reproducible
64-bit payload (0.000244 bpp). Triple-hybrid marked-image PSNR was
48.6566 ± 0.2439 dB, predictor MAE was 4.6288 ± 1.9358 gray levels, embedding
runtime was 0.6850 ± 0.0880 seconds, extraction runtime was 0.6818 ± 0.0586
seconds, and sidecar storage was 1,810.2 ± 639.7 bytes. Its reported
capacity (27,390.5 ± 12,611.4 safe zero-error positions) is not a maximum
payload sweep. For all four configurations, mean ± sample SD across the 19
images was:

| Configuration | Safe zero-error positions | Marked PSNR (dB) | Predictor MAE (gray levels) | Embed (s) | Extract (s) | Sidecar (bytes) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| RP-only | 26,191.7 ± 11,905.6 | 48.6336 ± 0.2279 | 5.1050 ± 2.1186 | 0.4379 ± 0.0373 | 0.4905 ± 0.0448 | 1,518.9 ± 666.6 |
| CNN-only | 27,259.4 ± 12,512.3 | 48.6539 ± 0.2417 | 4.6321 ± 1.9345 | 0.6824 ± 0.1651 | 0.6704 ± 0.0597 | 1,628.7 ± 605.1 |
| Transformer-only | 27,248.2 ± 12,424.5 | 48.6533 ± 0.2397 | 4.6498 ± 1.9068 | 1.2861 ± 0.0738 | 1.0454 ± 0.0505 | 1,647.1 ± 565.5 |
| Triple-hybrid router | 27,390.5 ± 12,611.4 | 48.6566 ± 0.2439 | 4.6288 ± 1.9358 | 0.6850 ± 0.0880 | 0.6818 ± 0.0586 | 1,810.2 ± 639.7 |

The triple-hybrid sidecar averaged 1.0435% ± 0.4018% of the marked PNG
storage. The sidecar is required; its storage overhead is separate from
marked-image PSNR. Predictor MAE is not an RDH distortion metric, and the
safe zero-error count is not a maximum payload result.

Clean-v2 artifacts are saved under `checkpoints/clean_v2/` and
`outputs/clean_v2_experiment/`. The experiment manifests, per-image CSVs, and
aggregate CSVs are the sources of record for these values.

## Reversible Data Hiding Status

Prediction MAE measures predictor quality, not RDH capacity or marked-image
distortion. The experimental PEE implementation is in `src/rdh/pee.py` and
returns a marked grayscale image plus a required, compressed `.npz` sidecar.
The sidecar stores pass lengths, dimensions, predictor/router configuration,
model-weight fingerprints, cover/payload checksums, and per-pass overflow
indices. Use `save_marked_image` for lossless PNG storage and `save_sidecar`
for the required auxiliary file; both refuse to overwrite existing files. The codec embeds
Cross then Dot and extracts/restores Dot then Cross, excluding the outer
one-pixel border.

This sidecar-based codec is not self-contained and has explicit storage
overhead. Its PEE mapping is a project implementation; no baseline-paper PDF
or source is present in `paper/`, so it must not be described as an exact
reproduction of Zhao et al. Per-image capacity, embedding rate, marked-image
PSNR, runtime, predictor MAE, and sidecar size are available through
`compute_rdh_metrics` in `src/rdh/pee.py`. Exact recovery claims require
passing round-trip tests.

There is no standalone embed/extract CLI. Use the Python API:

```python
from src.data.loader import load_grayscale_image
from src.rdh.pee import (
    embed_payload, extract_payload, load_sidecar, make_rp_predictor,
    save_marked_image, save_sidecar,
)

cover, _ = load_grayscale_image("cover.tiff")
predictor, predictor_metadata = make_rp_predictor()
encoded = embed_payload(cover, payload_bytes, predictor, predictor_metadata)
save_marked_image(encoded.marked_image, "marked.png")
save_sidecar(encoded.sidecar, "marked.rdh.npz")

decoded = extract_payload(
    load_grayscale_image("marked.png")[0],
    load_sidecar("marked.rdh.npz"),
    predictor,
    predictor_metadata,
)
```

Run the new USC-SIPI benchmark with the existing checkpoints and thresholds:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rdh_benchmark_v1.py
```

It writes only to `outputs/rdh_benchmark_v1/` and refuses to overwrite that
directory.

### Running Unit Tests
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
