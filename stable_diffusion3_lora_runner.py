import os
import subprocess
import torch
import gc
from huggingface_hub import snapshot_download
from diffusers import DiffusionPipeline
from huggingface_hub import login

def install_dependencies():
    """Installs necessary libraries and packages for the project."""
    print("Starting dependency installation...")
    try:
        subprocess.run(["pip", "install", "-q", "-U", "git+https://github.com/huggingface/diffusers"], check=True)
        subprocess.run(["pip", "install", "-q", "-U", "transformers", "accelerate", "wandb", "bitsandbytes", "peft"], check=True)
        print("All dependencies installed successfully.")
    except subprocess.CalledProcessError as error:
        print(f"Failed to install dependencies: {error}")

def login_to_huggingface(api_key):
    """Logs in to the Hugging Face platform."""
    print("Logging in to Hugging Face...")
    try:
        login(token=api_key)
        print("Successfully logged into Hugging Face.")
    except Exception as error:
        print(f"Login failed: {error}")

def clone_diffusers_repository():
    """Clones the Diffusers GitHub repository for stable diffusion examples."""
    print("Cloning Diffusers repository...")
    try:
        repo_dir = "diffusers"
        if os.path.exists(repo_dir):
            print(f"Repository {repo_dir} already exists. Removing to re-clone.")
            subprocess.run(["rm", "-rf", repo_dir], check=True)
        
        subprocess.run(["git", "clone", "https://github.com/huggingface/diffusers"], check=True)
        os.chdir("diffusers/examples/research_projects/sd3_lora_colab")
        print(f"Changed working directory to: {os.getcwd()}")
        print("Repository cloned and ready.")
    except subprocess.CalledProcessError as error:
        print(f"Failed to clone repository: {error}")
    except FileNotFoundError as error:
        print(f"Directory change failed: {error}")

def download_data():
    """Downloads and organizes the dataset needed for training."""
    print("Downloading dataset...")
    try:
        local_directory = "./outfits"
        snapshot_download(
            "manojkumarmaharana/outfits",
            local_dir=local_directory, repo_type="dataset",
            ignore_patterns=".gitattributes",
        )
        subprocess.run(["rm", "-rf", "outfits/.huggingface"], check=True)
        print("Dataset downloaded and organized.")
    except subprocess.CalledProcessError as error:
        print(f"Dataset download error: {error}")

def compute_model_embeddings():
    """Computes embeddings for model training."""
    print("Running embedding computation...")
    try:
        subprocess.run(["python", "compute_embeddings.py"], check=True)
        print("Embedding computation successful.")
    except subprocess.CalledProcessError as error:
        print(f"Embedding computation failed: {error}")

def clear_memory():
    """Clears CUDA memory cache and triggers garbage collection."""
    print("Clearing memory...")
    torch.cuda.empty_cache()
    gc.collect()
    print("Memory has been cleared.")

def train_diffusion_model():
    """Trains the model with specified parameters."""
    print("Starting model training...")
    try:
        # Ensure `train.py` has execute permissions in the home directory
        home_directory = os.path.expanduser("~")
        script_location = os.path.join(home_directory, "train.py")
        if not os.access(script_location, os.X_OK):
            print(f"Making {script_location} executable.")
            subprocess.run(["chmod", "+x", script_location], check=True)
        else:
            print(f"{script_location} is already executable.")

        result = subprocess.run([
            "accelerate", "launch", script_location,
            "--pretrained_model_name_or_path=stabilityai/stable-diffusion-3-medium-diffusers",
            "--instance_data_dir=outfits",
            "--data_df_path=sample_embeddings.parquet",
            "--output_dir=trained-sd3-lora-miniature",
            "--mixed_precision=fp16",
            "--instance_prompt=photos of trendy genz outfits",
            "--resolution=1024",
            "--train_batch_size=1",
            "--gradient_accumulation_steps=4",
            "--gradient_checkpointing",
            "--use_8bit_adam",
            "--learning_rate=1e-4",
            "--report_to=wandb",
            "--lr_scheduler=constant",
            "--lr_warmup_steps=0",
            "--max_train_steps=10",
            "--seed=0",
            "--checkpointing_steps=100",
            "--push_to_hub",
            "--hub_model_id=manojkumarmaharana/sd3_finetuned_shecodes",
        ], check=True, capture_output=True, text=True)
        print("Model training completed successfully.")
    except subprocess.CalledProcessError as error:
        print(f"Model training error: {error}")
        print(f"stdout: {error.stdout}")
        print(f"stderr: {error.stderr}")

def perform_inference():
    """Runs inference on the trained model to generate images."""
    print("Starting inference...")
    try:
        if not os.path.exists("trained-sd3-lora-miniature"):
            raise FileNotFoundError("Training output directory not found. Ensure training was completed successfully.")

        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16
        )
        lora_output_path = "trained-sd3-lora-miniature"
        pipeline.load_lora_weights(lora_output_path)
        pipeline.enable_sequential_cpu_offload()

        # Generate and save images
        description = "Beige-coloured & black regular wrap top, Animal printed, V-neck, three-quarter,regular sleeves"
        for i in range(1, 6):
            image = pipeline(description).images[0]
            image.save(f"animal_print_{i}.png")
        print("Inference complete. Images saved as 'animal_print_X.png'.")
    except Exception as error:
        print(f"Inference error: {error}")

def check_cuda_device():
    """Checks if CUDA is available and selects the appropriate device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

if __name__ == "__main__":
    print("Initializing...")
    install_dependencies()
    
    api_key = "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"
    login_to_huggingface(api_key)

    check_cuda_device()
    
    # Uncomment following lines to enable each step
    # download_data()
    # compute_model_embeddings()
    # clear_memory()
    # train_diffusion_model()
    # clear_memory()
    perform_inference()
