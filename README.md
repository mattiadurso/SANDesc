# 
<p align="center">
  <h1 align="center">SANDesc: A Streamlined Attention-based Network for Descriptor Extraction</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=9FjTo3YAAAAJ&hl=en">Mattia D'Urso</a>
    ·
    <a href="https://scholar.google.com/citations?user=1JwKYK8AAAAJ&hl=en&oi=ao">Emanuele Santellani</a>
    ·
    <a href="https://scholar.google.com/citations?user=6uZVF04AAAAJ&hl=en">Christian Sormann</a>
    <br>
    <a href="https://scholar.google.com/citations?user=DA3nSvgAAAAJ&hl=en">Mattia Rossi</a>
    ·
    <a href="https://scholar.google.com/citations?user=-uuEU_wAAAAJ&hl=en">Andreas Kuhn</a>
    ·
    <a href="https://scholar.google.com/citations?user=M0boL5kAAAAJ&hl=en">Friedrich Fraundorfer</a>
  </p>
  <h2 align="center">
    <p>3DV 2026</p>
    <a href="https://arxiv.org/abs/2601.13126" align="center">Paper</a> | 
    <a href="https://mattiadurso.github.io/sandesc_3dv26/static/pdfs/SANDesc___3DV_SM.pdf" align="center">Supplementary Material</a> | 
    <a href="https://mattiadurso.github.io/sandesc_3dv26/" align="center"> Project Page</a>
  </h2>
</p>

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mattiadurso/sandesc.git
cd sandesc
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate sandesc
```

3. Setup paths:
```bash
python setup_paths.py
```

## Training

### Quick Start

Start training with default configuration:
```bash
python train_sandesc.py
```

Resume from checkpoint:
```bash
python train_sandesc.py resume_from=path/to/checkpoint.pth
```

### Configuration

Training parameters are configured using Hydra configs in the `configs/` directory. Key settings include:

- **Model**: Architecture settings (channels, attention, skip@inproceedings{durso2026sandesc,
        title={A Streamlined Attention-based Network for Descriptor Extraction},
        author={D'Urso, Mattia and Santellani, Emanuele and Sormann, Christian and Rossi, Mattia and Kuhn, Andreas and Fraundorfer, Friedrich},
        booktitle={2026 International Conference on 3D Vision (3DV)},
        year={2026},
        organization={IEEE Computer Society}
      } connections)
- **Training**: Batch size, learning rate, iterations, dataset selection
- **Loss**: Triplet loss margin, ratio, negative mining parameters

### Features

- **Hard Negative Mining**: Dynamic triplet selection for effective learning
- **Mixed Precision**: AMP support for faster training
- **Multiple Datasets**: Support for MegaDepth, IMB, synthetic data
- **Comprehensive Logging**: WandB integration
- **Reproducible**: Deterministic algorithms and seed management

## Project Structure

```
sandesc/
├── configs/               # Hydra configuration files
├── datasets/              # Dataset loaders (MegaDepth, IMB, TerraSky3D etc.)
├── model/                 # SANDesc network architecture
├── losses/                # Triplet loss implementation
├── train_utils/           # Training utilities and evaluation
├── train_sandesc.py       # Main training script
└── README.md
```

## Citation

```bibtex
@inproceedings{durso2026sandesc,
        title={A Streamlined Attention-based Network for Descriptor Extraction},
        author={D'Urso, Mattia and Santellani, Emanuele and Sormann, Christian and Rossi, Mattia and Kuhn, Andreas and Fraundorfer, Friedrich},
        booktitle={2026 International Conference on 3D Vision (3DV)},
        year={2026},
        organization={IEEE Computer Society}
      }
```
