#!/bin/bash -l
#SBATCH -J MATLAB          # job name
#SBATCH -o ./job.out.%j    # standard out file
#SBATCH -e ./job.err.%j    # standard err file
#SBATCH -D ./  
#SBATCH --nodes=1            # work directory
#SBATCH --ntasks-per-node=50         # launch job on a single core
# #SBATCH --cpus-per-task=36  #   on a shared node
# #SBATCH --mem=10000
#SBATCH --time=23:00:00


module load matlab
srun matlab -nodisplay -nosplash -nodesktop -noFigureWindows -r 'run("CSP_slurmtest_nomov_SBA.m")'