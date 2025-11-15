#!/bin/bash

ls /u/user/ab0633/anaconda3/etc/profile.d/conda.sh
export PATH=~/anaconda3/bin:$PATH
source /u/user/ab0633/anaconda3/etc/profile.d/conda.sh

conda activate analysis

cd /u/user/ab0633/analysis/analysis_WZG/condor

python3 All_pre_selec_ver3_WZG.py $1 $2  

