# Weakly Supervised Multiple Instance Learning for Whale Call Detection and Localization in Long-Duration Passive Acoustic Monitoring

This repository contains code for training and evaluating multiple instance learning (MIL) models for detecting and localizing whale calls in underwater acoustic recordings. This implementation is based on research for detecting Antarctic blue whale and fin whale vocalizations.

## Features

- Preprocessing pipeline for extracting acoustic features from raw audio
- MIL architecture with attention mechanisms for call detection and localization
- Multi-GPU training with cross-validation
- Evaluation tools for detailed performance analysis

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.3+
- CUDA 12.4+ (for GPU acceleration)

### Setup

1. Clone this repository:
   ```bash
   https://github.com/Ragib-Amin-Nihal/DSMIL-Loc.git
   cd DSMIL-Loc
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Data Preparation: [dataprepare.md](dataprepare.md)

Before training models, you'll need to obtain and preprocess the raw acoustic data. Please refer to [dataprepare.md](dataprepare.md) for detailed instructions on:

- Obtaining the dataset from the Australian Antarctic Data Center
- Setting up S3 credentials and downloading the data
- Preprocessing the raw audio into "bags" for training

## Usage

### 1. Data Preprocessing

Process raw audio files into "bags" of instance spectrograms:

```bash
python scripts/create_bags.py \
    --data-root /path/to/raw/data \
    --output-root /path/to/output \
    --site-years kerguelen2014 kerguelen2015 casey2014 \
    --bag-durations 300 \
    --instance-durations 15
```

### 2. Training

Run training with cross-validation:

```bash
python scripts/train_mil.py \
    --config configs/default_config.yml \
    --data-dir /path/to/processed/data \
    --output-dir /path/to/results
```


### 3. Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate.py \
    --model-path /path/to/model/checkpoint.pt \
    --data-root /path/to/processed/data \
    --output-dir /path/to/evaluation/results
```

## Model Architecture

The model architecture is based on the `ImprovedLocalizationMILNet` class, which includes:

- A spectrogram encoder with residual connections
- A temporal feature encoder
- Multi-head self-attention for instance aggregation
- Positional encoding for sequence awareness
- Gated attention mechanism for localization

## Configuration

Configuration options are stored in YAML files in the `configs/` directory. The default configuration includes:

- Model parameters (feature dimensions, number of attention heads)
- Training parameters (batch size, learning rate, number of epochs)
- Early stopping settings
- Paths for data, checkpoints, and results

## Citation

If you use this code in your research, please cite:


## License

This project is licensed under the MIT License - see the LICENSE file for details.
