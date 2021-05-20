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

submit_cmd = "python -m aml_tools.aml_submit --input_dir . --output_dir {} --num_nodes 1 --exp_name t-lqing " \
             "--config_yaml ~/.azureml/{}.yaml --cmd \"{}\""

output_dir = 't-lqing/experiments/captioning/pruning_{}_{}/{}'
job_cmd = "oscar/run_captioning_pruning.py \
    --model_name_or_path experiments/captioning/Oscar_B+pruning_coef_2/checkpoint-0-1000 \
    --do_train --do_test --do_lower_case --add_od_labels --tie_weights --freeze_embedding \
    --label_smoothing 0.1 --drop_worst_ratio 0.2 --drop_worst_after 20000 --output_dir output/ \
    --evaluate_during_training --save_steps 5000 \
    --self_slimming --inter_slimming \
    --prune_before_train --self_pruning_method=layerwise --inter_pruning_method=global \
    --pruning_strategy {} --pruning_ratio {} --seed {} \
"
n_repeat = 3
for seed in range(n_repeat):
    for pruning_ratio in [0.2, 0.4, 0.6, 0.8]:
        for pruning_strategy in ['small', 'large', 'random']:
            resolved_output_dir = output_dir.format(pruning_strategy, pruning_ratio, seed)
            resolved_job_cmd = job_cmd.format(pruning_strategy, pruning_ratio, seed)
            resolved_submit_cmd = submit_cmd.format(resolved_output_dir, ws_config, resolved_job_cmd)
            print(resolved_submit_cmd)
            os.system(resolved_submit_cmd)
            # exit()
            
