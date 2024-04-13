#!/bin/bash
#SBATCH --job-name=a3_lora
#SBATCH --account=def-sponsor00
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M

# Set the path to data and model dir
PROJECT_HOME_DIR=/home/$USER/projects/def-sponsor00
PROJECT_DIR=$PROJECT_HOME_DIR/$USER
LORA_SAVE_DIR=$PROJECT_DIR/model/pythia_lora
MODEL_DIR=$PROJECT_DIR/data/ift6289_a3/model/pythia-410m/EleutherAI/pythia-410m
REPO_DIR=$SCRATCH/litgpt  # Change this to your repo directory or leave it as is and put your folder in $SCRATCH
CACHE_DIR=/home/$USER/scratch/cache

# Load necessary modules
module load arrow rust

# Generate your virtual environment in $SLURM_TMPDIR, don't change this line
virtualenv --no-download "${SLURM_TMPDIR}"/my_env && source "${SLURM_TMPDIR}"/my_env/bin/activate

# Install packages on the virtualenv, please always have the no-index argument
mkdir -p $CACHE_DIR
cd "$REPO_DIR" || exit
pip install $PROJECT_HOME_DIR/torch-2.2.2+computecanada-cp310-cp310-linux_x86_64.whl
XDG_CACHE_HOME=$CACHE_DIR pip install -e '.[all]'

# Create the lora save directory
mkdir -p $LORA_SAVE_DIR

# Setup wandb (requires WANDB_API_KEY setup in your ~/.bash_profile)
wandb login $WANDB_API_KEY

# Uncomment the following line and add your Huggingface Token here
# Generate or retrieve your Huggingface Token at https://huggingface.co/settings/tokens
# export HF_TOKEN=[YOUR_HF_TOKEN]

# LoRA instruction tuning
# Before finetuning, please run torch.cuda.is_bf16_supported(). If it returns False, set --precision fp16-true
XDG_CACHE_HOME=$CACHE_DIR litgpt finetune lora \
  --data LIMA \
  --checkpoint_dir $MODEL_DIR \
  --out_dir $LORA_SAVE_DIR \
  --train.max_seq_length 512 \
  --train.epochs 15 \
  --train.lr_warmup_steps 10 \
  --precision 16-mixed \
  --lora_r 8 \
  --lora_alpha 16 \
  --eval.interval 50 \
  --logger wandb