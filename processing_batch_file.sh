#!/bin/bash
#SBATCH -o /data/hpcdata/users/rablack75/power_HPCFlash/out/%a.out                  # output file 
#SBATCH --error=/data/hpcdata/users/rablack75/power_HPCFlash/out/%a.err             # error file
#SBATCH -J BurstPowFlash                                                            # name the job 
#SBATCH --mem=160gb                                                                 # the memory for each node
#SBATCH --time=12:00:00                                                             # length of job
#SBATCH --mail-type=begin,end,fail                                                  # email failure messages 
#SBATCH --mail-user=rablack75@bas.ac.uk                                            # to my email 
#SBATCH --partition=medium                                                          # which nodes to use
#SBATCH --account=medium                                                            # on which account
#SBATCH --cpus-per-task=1
#SBATCH --wait                                                                      # wait until this script has finished running before starting any other process (e.g. in an outer bash script)                                     

source /data/hpcdata/users/rablack75/burstenv/bin/activate                          # activate environment

date=$(date -d "$currdate + ${SLURM_ARRAY_TASK_ID}day" +%Y%m%d)                     # setting the day using the ${SLURM_ARRAY_TASK_ID} as the index

python /data/hpcdata/users/rablack75/power_HPCFlash/power_bands.py ${date}    