#!/bin/bash

#block(name=guanghan, threads=1, memory=10000, subtasks=1, gpu=true, hours=1)
    python main_gcn.py --non_local --epoch 30 --num_layers 64
    #python Bundle_Adjustment_motion_refine_optimization_new.py people72.json
    # can check the status and available resources using commands: `qstat`, `qinfo`.
    sleep 5
    echo "this block ONLY starts after the previous block finishes. Remember to set gpu=true."

