#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --cpus-per-task=48
#SBATCH --time=02:30:00
#SBATCH -o /data/mfulghieri/quokka/my_worksite/myAnalysis/DMZeldovich/outputs/log.out
#SBATCH -e /data/mfulghieri/quokka/my_worksite/myAnalysis/DMZeldovich/outputs/err.err


DIR="/data/mfulghieri/quokka/src/problems/DMZeldovich"
EXE="/data/mfulghieri/quokka/my_worksite/myAnalysis/DMZeldovich/analyze_cosmoDM.py"



python3 ${EXE} \
       --plotfiles "/data/mfulghieri/quokka/outputs/DMZeldovich/DMZeldovich_${SLURM_JOB_ID}/plt*" \
       --save   "/data/mfulghieri/quokka/outputs/DMZeldovich/DMZeldovich_${SLURM_JOB_ID}/analysis"