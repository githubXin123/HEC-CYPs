# HEC-CYPs: Hybrid Encoder-Classifier for CYP450 Inhibition Prediction

A deep learning framework for predicting CYP450 enzyme inhibition using ESM-2 (protein language model) and UniMol (3D molecular representation).

## Overview

HEC-CYPs predicts whether a compound inhibits any of the 7 major CYP450 isoforms by combining a pre-trained protein language model (ESM-2) with a 3D molecular encoder (UniMol) through a cross-attention mechanism.


## Model Architecture

- **UniMol**: 15-layer Transformer encoder with pair features, pre-trained on 77M molecules
- **ESM-2**: ESM-2 650M parameter model, fine-tuned on eukaryotic P450 sequences
- **Cross-Attention**: Molecule embedding attends to protein embedding
- **MLP**: 512→128→32→32→2 classifier head

## Directory Structure

```
├── model.py                       # Model definition and helper functions
├── train_model.py                 # Training from scratch
├── finetune_model.py              # Fine-tuning from pre-trained checkpoint
├── test_model.py                  # Model evaluation on test set
├── pred_model.py                  # Prediction on new compounds
├── prep_data.py                   # Data preparation (3D conformer generation)
││
├── utils/                         # Core modules
│   ├── model_scripts.py           # Model architecture (EsmUnimolClassifier)
│   ├── featurizer.py              # 3D conformer & UniMol dataset pipeline
│   ├── configurator.py            # JSON config loader
│   ├── benchmark.py               # Evaluation metrics
│   ├── pytorchtools.py            # Early stopping, seed, timing
│   └── globals.py                 # Thread-safe mutex
│
├── template/                      # JSON config templates
│   ├── data_preparation.json
│   ├── model_training.json
│   ├── model_finetuning.json
│   ├── model_testing.json
│   └── model_prediction.json
│
└── datasets/                      # Training/testing data (user-provided)
```

## Setup

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- Conda (recommended for environment management)

### Installation

```bash
# Create conda environment
conda create -n hec_cyps python=3.11
conda activate hec_cyps

# Install PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install numpy pandas scikit-learn rdkit tqdm tensorboardX openpyxl
pip install fair-esm

# Install UniMol from source
git clone https://github.com/dptech-corp/Uni-Mol.git
cd Uni-Mol/unimol
pip install -e .
cd ../..

# Install pySMASH (for substructure analysis)
pip install pysmash
```

### Pre-trained Weights

The following pre-trained weights are required. Download and update the paths in `utils/model_scripts.py`:

| Weight | Source | Path reference |
|--------|--------|----------------|
| UniMol pre-trained (`mol_pre_no_h_220816.pt`) | UniMol release | `model_pts/mol_pre_no_h_220816.pt` |
| ESM-2 650M (`esm2_t33_650M_UR50D.pt`) | FAIR | `model_pts/esm2_t33_650M_UR50D.pt` |
| P450 fine-tuned ESM-2 (`p450_eukaryota_esm_2.pth`) | Domain adaptation | `model_pts/p450_eukaryota_esm_2.pth` |

## Usage

All scripts use JSON configuration templates from the `template/` directory.

### 1. Data Preparation

Generate 3D conformers and prepare data loaders:

```bash
python prep_data.py -t template/data_preparation.json
```

The template specifies isoforms, task name, and number of worker threads.

### 2. Training

Train a model from scratch:

```bash
python train_model.py -t template/model_training.json
```

### 3. Fine-Tuning

Fine-tune from a pre-trained checkpoint:

```bash
python finetune_model.py -t template/model_finetuning.json
```

The `freeze_mode` parameter controls parameter freezing:
- `'all'`: freeze everything (original fixed behavior)
- `'encoders'`: freeze UniMol + ESM-2, train attention + MLP
- `'none'`: full parameter fine-tuning

### 4. Testing

Evaluate on a test set:

```bash
python test_model.py -t template/model_testing.json
```

### 5. Prediction

Predict inhibition for new compounds:

```bash
python pred_model.py -t template/model_prediction.json
```


## Data Format

Each isoform's data should be organized under `datasets/` as follows:


## License

MIT License

## Citation

If you use HEC-CYPs in your research, please cite the associated publication.


