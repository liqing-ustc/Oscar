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

task = 'vqa'
submit_cmd = "python -m aml_tools.aml_submit --input_dir . --output_dir {} --num_nodes 1 --exp_name liqing-{} " \
             "--config_yaml ~/.azureml/{}.yaml --cmd \"{}\""

output_dir = 't-lqing/experiments/vqa/pruning_checkpoint-{}/pruning_{}_{}/{}'
job_cmd = "oscar/run_vqa_pruning.py \
        --do_train --do_lower_case \
        --model_name_or_path=experiments/vqa/Oscar_B+pruning_coef/checkpoint-{} \
        --evaluate_during_training --save_steps 1000 \
        --self_slimming --inter_slimming --prune_before_train \
        --pruning_strategy {} --pruning_ratio {} --seed {}\
"
for ck in ['0-2000', '1-4000', '4-10000']:
    n_repeat = 1
    for seed in range(n_repeat):
        for pruning_ratio in [0.2, 0.4, 0.6, 0.8]:
            for pruning_strategy in ['small', 'large', 'random']:
                resolved_output_dir = output_dir.format(ck, pruning_strategy, pruning_ratio, seed)
                resolved_job_cmd = job_cmd.format(ck, pruning_strategy, pruning_ratio, seed)
                resolved_submit_cmd = submit_cmd.format(resolved_output_dir, task, ws_config, resolved_job_cmd)
                print(resolved_submit_cmd)
                os.system(resolved_submit_cmd)
                # exit()
            
