# Stable Diffusion 3 with DreamBooth LoRA

This repository provides the code necessary for training and performing inference with Stable Diffusion 3, leveraging both DreamBooth and LoRA (Low-Rank Adaptation) for image generation. The purpose of this project is to create images from text prompts and to fine-tune models using custom datasets.

## Table of Contents
- [Overview](#overview)
- [Setup Instructions](#setup-instructions)
- [How to Use](#how-to-use)
  - [Embedding Computation](#embedding-computation)
  - [Model Training](#model-training)
  - [Running Inference](#running-inference)
- [File Descriptions](#file-descriptions)
- [Requirements](#requirements)

## Overview

This project integrates Stable Diffusion 3, DreamBooth, and LoRA to generate high-quality images from text-based prompts. The repository includes three main scripts: `compute_embeddings.py`, `train.py`, and `stable_diffusion3_lora_runner.py`. A `requirements.txt` file is also included for dependency installation.

## Setup Instructions

First, clone this repository and navigate to the project directory:

```sh
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
```

Next, install the required dependencies using:

```sh
pip install -r requirements.txt
```

## How to Use

### Embedding Computation

The `compute_embeddings.py` script calculates the embeddings for a given set of images and saves the output as a Parquet file.

- Ensure that the images are in a folder, with each image having a corresponding caption in a text file (e.g., `image1.jpg` and `image1.txt`).

**Arguments**:
- `--prompt`: The text prompt describing the images (default: "photos of trendy genz outfits").
- `--max_sequence_length`: Maximum token sequence length for embeddings (default: 77).
- `--local_data_dir`: Directory path containing images (default: "outfits").
- `--output_path`: Location to save the Parquet file with the computed embeddings (default: "sample_embeddings.parquet").

### Model Training

The `train.py` script is used to train a model using the previously generated embeddings. The trained model is saved upon completion.

- Ensure the instance data and embeddings file (`sample_embeddings.parquet`) are available before starting the training process.

**Arguments**:
- `--pretrained_model_name_or_path`: Path to the pretrained model or the Hugging Face model identifier.
- `--instance_data_dir`: Directory path containing instance images.
- `--data_df_path`: Path to the Parquet file with embeddings.
- `--output_dir`: Directory to save the trained model.
- `--mixed_precision`: Precision type for training (e.g., `fp16`).
- `--instance_prompt`: Prompt used during training.

Additional arguments for customization are available within the script.

### Running Inference

The `stable_diffusion3_lora_runner.py` script is used for both installing dependencies, logging into Hugging Face, and running the inference process.

- Make sure you have the Hugging Face API key for authentication.

Run the inference with the following command:

```sh
python stable_diffusion3_lora_runner.py
```

**Key Functions**:
- `install_dependencies()`: Installs all required packages.
- `huggingface_login(api_key)`: Logs into Hugging Face.
- `clone_diffusers_repo()`: Clones the Diffusers GitHub repository.
- `compute_embeddings()`: Computes embeddings.
- `train_model()`: Trains the model.
- `run_inference()`: Runs inference and saves the generated images.

## File Descriptions
- `compute_embeddings.py`: Script for calculating image embeddings.
- `train.py`: Script for model training.
- `stable_diffusion3_lora_runner.py`: Script handling the installation, Hugging Face login, training, and inference.
- `requirements.txt`: Contains the list of Python packages needed.

## Requirements
Install all required dependencies with:

```sh
pip install -r requirements.txt
