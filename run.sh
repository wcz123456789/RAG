#!/bin/bash
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --job-name=rag_Qwen2.5-1.5B               # create a short name for your job
#SBATCH --nodes=1                                 # node count
#SBATCH --gpus=1                                  # number of GPUs per node(only valid under large/normal partition)
#SBATCH --time=0:10:00                           # total run time limit (HH:MM:SS)
#SBATCH --partition=normal                        # partition(large/normal/cpu) where you submit
#SBATCH --account=mscaisuperpod                   # only require for multiple projects

# 设置 CUDA 环境
export CUDA_HOME=/cm/shared/apps/nvhpc/23.11/Linux_x86_64/23.11/compilers
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module purge                     # clear environment modules inherited from submission
module load Anaconda3/2023.09-0  # load the exact modules required
# 在 module load 之后、conda activate 之前添加
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate rag
# /home/cwangeu/.conda/envs/rag/bin/python run.py --dense_retriever_lists m3e --recursive_answer --rephrase_context --rephrase_before_retrieve --answer_before_retrieve 
/home/cwangeu/.conda/envs/rag/bin/python ./evaluation/test_scores.py
