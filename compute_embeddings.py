import argparse
import glob
import hashlib
import os

import pandas as pd
import torch
from transformers import T5EncoderModel
from diffusers import StableDiffusion3Pipeline

# Constants
DEFAULT_PROMPT = "photos of trendy genz outfits"
DEFAULT_MAX_SEQ_LENGTH = 77
IMAGE_DIRECTORY = "outfits"
PARQUET_OUTPUT = "sample_embeddings.parquet"

def convert_bytes_to_gb(byte_count):
    """Convert bytes to gigabytes."""
    return byte_count / (1024 ** 3)

def compute_image_hash(image_file):
    """Generate SHA-256 hash for an image file."""
    with open(image_file, "rb") as file:
        file_data = file.read()
    return hashlib.sha256(file_data).hexdigest()

def initialize_pipeline():
    """Load and initialize the Stable Diffusion 3 pipeline with specified configurations."""
    model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
    encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder_3", load_in_8bit=True, device_map="auto")
    sd_pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id, text_encoder_3=encoder, transformer=None, vae=None, device_map="balanced"
    )
    return sd_pipeline

@torch.no_grad()
def generate_embeddings(sd_pipeline, prompt, description, max_len):
    """Generate embeddings for given prompt and caption."""
    (positive_embed, negative_embed, pooled_pos_embed, pooled_neg_embed) = sd_pipeline.encode_prompt(
        prompt=prompt, prompt_2=description, prompt_3=None, max_sequence_length=max_len
    )
    print(
        f"{positive_embed.shape=}, {negative_embed.shape=}, {pooled_pos_embed.shape=}, {pooled_neg_embed.shape=}"
    )
    max_alloc_memory = convert_bytes_to_gb(torch.cuda.max_memory_allocated())
    print(f"Max allocated memory: {max_alloc_memory:.3f} GB")
    return positive_embed, negative_embed, pooled_pos_embed, pooled_neg_embed

def execute(args):
    """Main execution function to load images, generate embeddings, and save them."""
    sd_pipeline = initialize_pipeline()
    image_files = [file for file in glob.glob(f"{args.image_directory}/*") if file.endswith(('.jpg', '.jpeg', '.png'))]
    embedding_data = []

    for img_file in image_files:
        img_hash = compute_image_hash(img_file)
        caption_file = os.path.splitext(img_file)[0] + ".txt"

        if os.path.exists(caption_file):
            with open(caption_file, "r") as file:
                caption_text = file.read().strip()
        else:
            caption_text = ""

        pos_embed, neg_embed, pooled_pos_embed, pooled_neg_embed = generate_embeddings(
            sd_pipeline, args.prompt, caption_text, args.max_sequence_length
        )

        embedding_data.append(
            (img_hash, caption_text, pos_embed, neg_embed, pooled_pos_embed, pooled_neg_embed)
        )

    embed_columns = ["pos_embed", "neg_embed", "pooled_pos_embed", "pooled_neg_embed"]
    embeddings_df = pd.DataFrame(
        embedding_data,
        columns=["image_hash", "caption"] + embed_columns,
    )

    for column in embed_columns:
        embeddings_df[column] = embeddings_df[column].apply(lambda x: x.cpu().numpy().flatten().tolist())

    embeddings_df.to_parquet(args.output_path)
    print(f"Embeddings successfully saved to {args.output_path}")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt for image generation.")
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help="Maximum sequence length for embedding computation.",
    )
    parser.add_argument(
        "--image_directory", type=str, default=IMAGE_DIRECTORY, help="Directory containing image files."
    )
    parser.add_argument("--output_path", type=str, default=PARQUET_OUTPUT, help="Path to output parquet file.")
    args = parser.parse_args()

    execute(args)
