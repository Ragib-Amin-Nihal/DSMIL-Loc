# Data Preparation Guide

This document outlines the complete process for obtaining and preparing the whale acoustic data for use with the Whale-MIL system.

## 1. Downloading the Dataset

### 1.1 Access the Australian Antarctic Data Center

1. Visit the Australian Antarctic Data Center website to request access to the dataset:
   [https://data.aad.gov.au/dataset/ba3d7e94-8e3e-4d09-8fe6-e9333a7730f9/download](https://data.aad.gov.au/dataset/ba3d7e94-8e3e-4d09-8fe6-e9333a7730f9/download)

2. Provide your email address as requested on the website.

### 1.2 Obtain S3 Credentials

You will receive an email containing S3 access information, including:
- Host address
- Account name
- Access key
- Secret key

### 1.3 Download the Dataset

1. Download and install an S3 client such as S3 Browser:
   [https://s3browser.com/download.aspx](https://s3browser.com/download.aspx)

2. For detailed instructions on using S3 clients with the Antarctic Data Center, refer to:
   [https://data.aad.gov.au/about/help-and-resources/download-guide](https://data.aad.gov.au/about/help-and-resources/download-guide)

3. Configure your S3 client with the provided credentials and download the dataset.

### 1.4 Extract the Dataset

If the downloaded data is in compressed format (ZIP), extract it to a location with sufficient storage space.

## 2. Dataset Structure

After extraction, you will have a folder named "AcousticTrends_BlueFinLibrary" with the following structure:

```
AcousticTrends_BlueFinLibrary/
├── BallenyIslands2015/
│   ├── wav/                  # Audio files in WAV format
│   └── *.selections.txt      # Annotation files
├── casey2014/
│   ├── wav/
│   └── *.selections.txt
├── kerguelen2005/
│   ├── wav/
│   └── *.selections.txt
├── kerguelen2014/
│   ├── wav/
│   └── *.selections.txt
└── [other site-years]/
```

Each site-year folder contains WAV audio recordings and corresponding annotation files (.selections.txt) that identify whale call events in the recordings.

## 3. Preprocessing: Creating Bags

The next step is to preprocess the raw audio data into "bags" for multiple instance learning. Each bag contains multiple audio instances with associated features and spectrograms.

### 3.1 Running the Preprocessing Script

Use the `create_bags.py` script to process the raw audio into bags:

```bash
python scripts/create_bags.py \
  --data-root /path/to/AcousticTrends_BlueFinLibrary \
  --output-root /path/to/output/directory \
  --site-years kerguelen2014 kerguelen2015 casey2014 \
  --bag-durations 300 \
  --instance-durations 15
```

Parameters:
- `--data-root`: Path to the raw dataset
- `--output-root`: Where to save the processed data
- `--site-years`: List of site-years to process
- `--bag-durations`: Duration of each bag in seconds (default: 300s = 5 minutes)
- `--instance-durations`: Duration of each instance in seconds (default: 15s)

For large datasets, this process may take several hours.

### 3.2 Processed Data Structure

After processing, your output directory will have the following structure:

```
output_directory/
└── results_bag300_inst15/      # Based on bag/instance duration
    └── bags/
        ├── kerguelen2014/
        │   ├── 20140101_000000/     # Bag ID (timestamp)
        │   │   ├── metadata.json
        │   │   ├── spectrograms/
        │   │   │   ├── instance_0000_spec.npy
        │   │   │   ├── instance_0001_spec.npy
        │   │   │   └── ...
        │   │   └── features/
        │   │       ├── instance_0000_features.json
        │   │       ├── instance_0001_features.json
        │   │       └── ...
        │   └── [more bags]
        ├── kerguelen2015/
        │   └── [bags]
        └── casey2014/
            └── [bags]
```

Each bag contains:
- `metadata.json`: Information about the bag, including whether it contains whale calls
- `spectrograms/`: Numpy arrays of spectrograms for each audio instance
- `features/`: JSON files containing extracted audio features for each instance

### 3.3 Bag Metadata

The `metadata.json` file contains important information about each bag:

```json
{
  "bag_id": "20140101_000000",
  "site_year": "kerguelen2014",
  "n_instances": 20,
  "start_time": "2014-01-01T00:00:00",
  "end_time": "2014-01-01T00:05:00",
  "duration": 300,
  "has_calls": true,
  "instances_with_calls": [
    {
      "instance_idx": 3,
      "labels": ["Bm-Ant-A"]
    },
    {
      "instance_idx": 4,
      "labels": ["Bm-Ant-A"]
    }
  ],
  "parameters": {
    "sampling_rate": 250,
    "nfft": 256,
    "hop_length": 16,
    "win_length": 256,
    "freq_range": [5, 124]
  }
}
```

The most important fields are:
- `has_calls`: Whether the bag contains any whale calls (used as the bag label)
- `instances_with_calls`: List of instances containing calls, with their indices and call types

## 4. Next Steps

Once preprocessing is complete, you can proceed to training using the processed data:

```bash
python scripts/train_mil.py \
  --data-root /path/to/output/directory \
  --output-root /path/to/training/results \
  --site-years kerguelen2014 kerguelen2015 casey2014 \
  --bag-durations 300 \
  --instance-durations 15 \
  --batch-size 64
```

This command will use the preprocessed data to train the MIL model.
