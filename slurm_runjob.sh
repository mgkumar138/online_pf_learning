#!/bin/bash
#SBATCH -J 1D_pf
#SBATCH -c 5
#SBATCH -t 10:00:00
### SBATCH -p kempner, gpu,pehlevan_gpu, seas_gpu, kempner_requeue
#SBATCH --account kempner_pehlevan_lab
### SBATCH -p sapphire,pehlevan, shared, seas_compute, serial_requeue
#SBATCH -p kempner_requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -o 1D_pf/_%A.%a.out
#SBATCH -e 1D_pf/_%A.%a.err
#SBATCH --array=0

# Activate conda before running this script.
eval "$(conda shell.bash hook)"
conda activate jax

# Construct the command with conditional variance_correction flag
<<<<<<< HEAD
CMD="python -u ./reward_org/1D/1D_track_td.py --episodes 10000 --lr 0.0001 --clr 0.001 --gamma 0.9 --seed ${SLURM_ARRAY_TASK_ID} "
=======
CMD="python -u ./reward_org/1D/1D_track_pg.py --episodes 20000 --lr 0.001 --clr 0.001 --gamma 0.9 --seed ${SLURM_ARRAY_TASK_ID} "
>>>>>>> 5692856 (all)

# Run the experiment script with the constructed command
echo $CMD
eval $CMD
