#! /home/qing/.virtualenvs/azure/bin/python
import os
import pdb
from datetime import datetime

ws_config = 'itp_acv' 
stamp = ''

if not stamp:
    stamp = datetime.now().strftime('%Y%m%d.%H%M%S')


output_dir = 't-lqing/output/oscar.{}'
submit_cmd = "python -m aml_tools.aml_submit --input_dir . --output_dir {} --num_nodes 1 --exp_name t-lqing " \
             "--config_yaml ~/.azureml/{}.yaml  --cmd \"{}\""
job_cmd = "oscar/run_captioning.py \
    --model_name_or_path models/pretrained_base/checkpoint-2000000 \
    --do_train \
    --do_test \
    --evaluate_during_training --save_steps 5000 \
    --do_lower_case \
    --add_od_labels \
    --learning_rate 3e-5 \
    --per_gpu_train_batch_size 64 \
    --num_train_epochs 60 \
    --tie_weights \
    --freeze_embedding \
    --label_smoothing 0.1 \
    --drop_worst_ratio 0.2 \
    --drop_worst_after 20000 \
    --output_dir output/ \
"
          

resolved_output_dir = output_dir.format(stamp)
resolved_submit_cmd = submit_cmd.format(resolved_output_dir, ws_config, job_cmd)
print(resolved_submit_cmd)
os.system(resolved_submit_cmd)
