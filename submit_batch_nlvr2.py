#! /home/qing/.virtualenvs/azure/bin/python
import os
import pdb
from datetime import datetime

# ws_config = 'itp_acv' 
# ws_config = 'vlp_cust' 
ws_config = 'vlp' 
# ws_config = 'objectdet_wu' 
stamp = ''

if not stamp:
    stamp = datetime.now().strftime('%Y%m%d.%H%M%S')

task = 'nlvr2'
submit_cmd = "python -m aml_tools.aml_submit --input_dir . --output_dir {} --num_nodes 1 --exp_name liqing-{} " \
             "--config_yaml ~/.azureml/{}.yaml --cmd \"{}\""

output_dir = 't-lqing/experiments/nlvr2/pruning_{}_{}_{}/{}'
job_cmd = "python oscar/run_nlvr_pruning.py \
    -j 4 --img_feature_dim 2054 --max_img_seq_length 40 --data_dir datasets/nlvr2 \
    --model_type bert --model_name_or_path models/pretrained_base/checkpoint-2000000 \
    --task_name nlvr --do_lower_case --max_seq_length 55 \
    --per_gpu_eval_batch_size 64 --per_gpu_train_batch_size 32 \
    --learning_rate 3e-05 --num_train_epochs 20 \
    --output_dir output --img_feature_type faster_r-cnn --data_label_type all --train_data_type all \
    --eval_data_type all --loss_type xe --save_epoch -1 --evaluate_during_training \
    --logging_steps -1 --drop_out 0.3 --do_train --weight_decay 0.05 --warmup_steps 10000 \
    --classifier mlp --cls_hidden_scale 3 --num_choice 2 --use_pair \
    --self_slimming --inter_slimming --l1_loss_coef 1e-4 \
    --pruning_strategy {} --pruning_ratio {} --pruning_steps {} --seed {}\
"

n_repeat = 1
for seed in range(n_repeat):
    for pruning_steps in [1000]:
        for pruning_ratio in [0.2, 0.4, 0.6, 0.8]:
            for pruning_strategy in ['small']:
                args = (pruning_strategy, pruning_ratio, pruning_steps, seed)
                resolved_output_dir = output_dir.format(*args)
                resolved_job_cmd = job_cmd.format(*args)
                resolved_submit_cmd = submit_cmd.format(resolved_output_dir, task, ws_config, resolved_job_cmd)
                print(resolved_submit_cmd)
                os.system(resolved_submit_cmd)
                # exit()
            
