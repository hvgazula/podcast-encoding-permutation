#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH -o './logs/%A.out'
#SBATCH -e './logs/%A.err'

if [[ "$HOSTNAME" == *"tiger"* ]]
then
    echo "It's tiger"
    module load anaconda
    source activate torch-env
elif [[ "$HOSTNAME" == *"della"* ]]
then
    echo "It's Della"
    module load anaconda
    source activate torch-env
else
    module load anacondapy
    source activate srm
fi

echo 'Requester:' $USER
echo 'Node:' $HOSTNAME
# echo 'Start time:' `date`
echo "$@"
if [[ -v SLURM_ARRAY_TASK_ID ]]
then
    python "$@"
else
    python "$@"
fi
# echo 'End time:' `date`
