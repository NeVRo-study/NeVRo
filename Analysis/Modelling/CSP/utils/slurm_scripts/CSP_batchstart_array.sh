#!/bin/bash -l
#SBATCH -J MATLAB          # job name
#SBATCH -o ./job.out.%A_%a    # standard out file
#SBATCH -e ./job.err.%A_%a    # standard err file
#SBATCH -D ./              # work directory
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # launch job on a single core
#SBATCH --array=26-43
#SBATCH --cpus-per-task=72  #   on a shared node
#SBATCH --mem=100000
#SBATCH --time=23:00:00

module load matlab
srun matlab -nodisplay -nosplash -nodesktop -noFigureWindows -r "run('CSP_slurm_array(${SLURM_ARRAY_TASK_ID}).m')"
