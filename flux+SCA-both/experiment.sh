#!/bin/bash
#SBATCH --job-name=fonts-train
#SBATCH -p suma_a100                   # A100 partition
#SBATCH -q a100_qos                    # QoS 정책 
#SBATCH --gres=gpu:2                   # 사용할 GPU 개수
#SBATCH --ntasks-per-node=1            # Accelerate/torchrun 1 task = 내부에서 프로세스 여러개 띄움
#SBATCH --cpus-per-task=8             # CPU core 할당 (GPU당 4코어 기준)
#SBATCH --mem=128G                     # CPU RAM
#SBATCH --time=2-00:00:00              # 2일
#SBATCH --output=/dev/null  # 버림
#SBATCH --error=/home/shaush/FonTS-main/flux+SCA-both/src/slurm_logs/fonts_train_%j.err   # 에러만

# (로그 디렉토리 생성)
mkdir -p /home/shaush/FonTS-main/flux+SCA-both/src/slurm_logs

echo "Job Start: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

source /home/shaush/miniconda3/etc/profile.d/conda.sh
conda activate mt5
echo "Conda environment 'mt5' activated."

cd /home/shaush/FonTS-main/flux+SCA-both/src

echo "Current directory: $(pwd)"
echo "Running accelerate launch..."

accelerate launch -m flux.train --config flux/config.yaml "$@"

echo "Job End: $(date)"