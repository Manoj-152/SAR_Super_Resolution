#!/bin/bash
source $HOME/.bashrc
conda activate /scratch/ms14845/point_env
cd $SCRATCH/SAR_Super_Resolution_2
python3 main.py ../Sen1-2