#!/bin/bash

#OAR -n FastDiag_models
#OAR -l /nodes=1/core=32,walltime=18:00:00
#OAR --stdout FastDiag_models.out
#OAR --stderr FastDiag_models.err
#OAR --project pr-gin5_aini

source ../environments/general_env/bin/activate
python3 compute_models.py