#! /home/qing/.virtualenvs/azure/bin/python
import os
import pdb
from datetime import datetime

ws_config = 'itp_acv' 
# ws_config = 'vlp_cust' 
# ws_config = 'vlp' 
stamp = ''

if not stamp:
    stamp = datetime.now().strftime('%Y%m%d.%H%M%S')

task = 'pretrain-vqa'
submit_cmd = "python -m aml_tools.aml_submit --input_dir . --output_dir {} --num_nodes 1 --exp_name liqing-{} " \
             "--config_yaml ~/.azureml/{}.yaml --cmd \"{}\""

output_dir = 't-lqing/experiments/pretrain/{}/0/vqa/checkpoint-{:07d}/'
job_cmd = "oscar/run_vqa_pruning.py --do_train --do_lower_case \
        --evaluate_during_training --save_steps 1000 \
        --self_slimming --inter_slimming \
        --seed 0\
        --model_name_or_path experiments/pretrain/{}/0/pretrain/checkpoint-{:07d} \
"

# n_repeat = 1
# for seed in range(n_repeat):
#     for pruning_steps in [4000, 3000, 2000, 1000]:
#         for pruning_ratio in [0.2, 0.4, 0.6, 0.8]:
#             for pruning_strategy in ['small', 'large', 'random']:
#                 args = (pruning_strategy, pruning_ratio, pruning_steps, seed)
#                 resolved_output_dir = output_dir.format(*args)
#                 resolved_job_cmd = job_cmd.format(*args)
#                 resolved_submit_cmd = submit_cmd.format(resolved_output_dir, task, ws_config, resolved_job_cmd)
#                 print(resolved_submit_cmd)
#                 os.system(resolved_submit_cmd)
#                 # exit()

n_repeat = 1
for seed in range(n_repeat):
    for pretrain_model in ['Oscar_B']:
        for ck in range(25, 19, -1):
            ck = int(ck * 1e4)
            args = (pretrain_model, ck)
            resolved_output_dir = output_dir.format(*args)
            resolved_job_cmd = job_cmd.format(*args)
            resolved_submit_cmd = submit_cmd.format(resolved_output_dir, task, ws_config, resolved_job_cmd)
            print(resolved_submit_cmd)
            os.system(resolved_submit_cmd)
            # exit()
            
