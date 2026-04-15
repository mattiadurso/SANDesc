<p align="center">
  <h1 align="center">SANDesc: A Streamlined Attention-based Network for Descriptor Extraction</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=9FjTo3YAAAAJ&hl=en">Mattia D'Urso</a><sup>1</sup>
    ·
    <a href="https://scholar.google.com/citations?user=1JwKYK8AAAAJ&hl=en&oi=ao">Emanuele Santellani</a><sup>1</sup>
    ·
    <a href="https://scholar.google.com/citations?user=6uZVF04AAAAJ&hl=en">Christian Sormann</a><sup>2</sup>
    <br>
    <a href="https://scholar.google.com/citations?user=DA3nSvgAAAAJ&hl=en">Mattia Rossi</a><sup>2</sup>
    ·
    <a href="https://scholar.google.com/citations?user=-uuEU_wAAAAJ&hl=en">Andreas Kuhn</a><sup>2</sup>
    ·
    <a href="https://scholar.google.com/citations?user=M0boL5kAAAAJ&hl=en">Friedrich Fraundorfer</a><sup>1</sup>
  </p>
  <p align="center">
    <sup>1</sup>Graz University of Technology · <sup>2</sup> Sony Europe
  </p>
  <h2 align="center">
    <p>3DV 2026</p>
    <a href="https://arxiv.org/abs/2601.13126" align="center">Paper</a> | 
    <a href="https://mattiadurso.github.io/sandesc/static/pdfs/SANDesc___3DV_SM.pdf" align="center">Supplementary Material</a> | 
    <a href="https://mattiadurso.github.io/sandesc/" align="center"> Project Page</a> |
    <a href="https://cloud.tugraz.at/index.php/s/4CtSeN4jgXRnLjB/download" align="center"> Graz4K</a>
  </h2>
</p>

<p align="center">
  <img src="assets/graz_clocktower/clock_sandesc.png" alt="ALIKED+SANDesc" width="400px">
  <img src="assets/graz_clocktower/clock_aliked.png" alt="ALIKED" width="400px">
  <br>
  <em>ALIKED+SANDesc (right) and ALIKED (left) on images from Clock Tower from the Graz 4K benchmark.</em>
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

## Demo
To make the demo easier to run, we run ALIKED on the demo images and saved its keypoints and descriptors. Since SANDesc is a descriptor module, it shares the same keypoints extracted by ALIKED. Then, only the descriptors are replaced and pose recomputed. Use `demo.ipynb` to visualize the demo.



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

- **Model**: Architecture settings:
  * `"ch_in"`: Input channel size.
  * `"kernel_size"`: Size of the convolutional kernel used.
  * `"activ"`: Activation function to use (`relu`, `gelu`).
  * `"norm"`: Normalization to use (`batch`, `instance`, `group`).
  * `"skip_connection"`: Whether to use a skip connection.
  * `"spatial_attention"`: Whether to use the CBAM module.
  * `"third_block"`: Whether to use the third block.


* **Training**: Traning keypoint detector, batch size, learning rate, iterations, and dataset selection.
* **Loss**: Triplet loss margin, ratio, and negative mining parameters.


## Project Structure

```
sandesc/
├── configs/               # Hydra configuration files
├── datasets/              # Dataset loaders (MegaDepth, IMB, TerraSky3D etc.)
├── losses/                # Triplet loss implementation
├── matcher/               # The matcher use for training
├── model/                 # SANDesc network architecture
├── utils/                 # Training utilities and evaluation
├── train_sandesc.py       # Main training script
└── README.md
```
## Testing

Please refer to [PoseBench](https://github.com/mattiadurso/PoseBench) for testing. PoseBench currently supports 7 different sparse feature extractors or matchers on 8 different benchmarks, including <a href="https://cloud.tugraz.at/index.php/s/4CtSeN4jgXRnLjB/download" align="center"> Graz4K</a>.

## Citation

```bibtex
@inproceedings{durso2026sandesc,
  title={A Streamlined Attention-based Network for Descriptor Extraction},
  author={Mattia D'Urso and Emanuele Santellani and Christian Sormann and Mattia Rossi and Andreas Kuhn and Friedrich Fraundorfer},
  booktitle={Thirteenth International Conference on 3D Vision},
  year={2026},
  url={https://openreview.net/forum?id=IM8t4BhDdG}
  }
```
