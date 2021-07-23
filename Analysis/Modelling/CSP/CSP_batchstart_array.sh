#!/bin/bash -l
#SBATCH -J MATLAB          # job name
#SBATCH -o ./job.out.%j    # standard out file
#SBATCH -e ./job.err.%j    # standard err file
#SBATCH -D ./              # work directory
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # launch job on a single core
#SBATCH --array=1-2
#SBATCH --cpus-per-task=72  #   on a shared node
# #SBATCH --mem=10000
#SBATCH --time=23:00:00

module load matlab
srun matlab -nodisplay -nosplash -nodesktop -noFigureWindows -r "run('CSP_slurm_array(${SLURM_ARRAY_TASK_ID}).m')"