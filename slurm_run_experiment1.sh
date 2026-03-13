
  


#!/bin/bash
#SBATCH --job-name=projects_ex1_gpu
#SBATCH --output=/home-mscluster/mgebreselassie/ProjectS/'Experiment 1_1'/slurm_%j.out
#SBATCH --error=/home-mscluster/mgebreselassie/ProjectS/'Experiment 1_1'/slurm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=bigbatch
#SBATCH --gres=gpu:1

cd /home-mscluster/mgebreselassie/ProjectS/'Experiment 1_1'


# --- Create/activate a venv (recommended) ---
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# --- Install GPU TensorFlow (official pip route) ---
pip install -r requirements-gpu.txt

# --- Sanity check: does TF see the GPU? ---
python check_gpu.py

# --- Run the new Experiment 1 sweep runner ---
python run_experiment1.py \
  --outputs Outputs \
  --nodes_list 5,10,15,20,30 \
  --degree_factors 1,2,3 \
  --dataset_sizes 500,1000,5000,10000 \
  --seeds 42,43
