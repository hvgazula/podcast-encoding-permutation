#!/bin/bash
#SBATCH --time=00:00:60
#SBATCH --mem=4GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH -o '%A.out'
#SBATCH -e '%A.err'

# if [[ "$HOSTNAME" == *"tiger"* ]]
# then
#     echo "It's tiger"
#     module load anaconda
#     source activate torch-env
# else
#     module load anacondapy
#     source activate srm
# fi

# if [[ "$HOSTNAME" == *"tiger"* ]]
# then
#     echo "It's tiger"
#     # module load anaconda
#     # source activate torch-env
# fi

echo 'Requester:' $USER
echo 'Node:' $HOSTNAME
# echo 'Start time:' `date`
echo "$@"
python "$@"
# echo 'End time:' `date`
