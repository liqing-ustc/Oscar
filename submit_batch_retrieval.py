#! /home/qing/.virtualenvs/azure/bin/python
import os
import pdb
from datetime import datetime

# ws_config = 'itp_acv' 
# ws_config = 'vlp_cust' 
# ws_config = 'vlp' 
ws_config = 'objectdet_wu' 
stamp = ''

if not stamp:
    stamp = datetime.now().strftime('%Y%m%d.%H%M%S')

task = 'coco_ir'
submit_cmd = "python -m aml_tools.aml_submit --input_dir . --output_dir {} --num_nodes 1 --exp_name liqing-{} " \
             "--config_yaml ~/.azureml/{}.yaml --cmd \"{}\""

output_dir = 't-lqing/experiments/coco_ir/pruning_{}_{}_{}/{}'
job_cmd = "python oscar/run_retrieval_pruning.py \
    --model_name_or_path datasets/coco_ir/model/base/checkpoint-1340000 \
    --do_train \
    --do_lower_case \
    --per_gpu_train_batch_size 16 \
    --learning_rate 0.00002 \
    --num_train_epochs 30 \
    --weight_decay 0.05 \
    --save_steps 5000 \
    --add_od_labels \
    --od_label_type vg \
    --max_seq_length 70 \
    --max_img_seq_length 70 \
    --output_dir output/ \
    --do_test \
    --do_eval \
    --test_split test \
    --num_captions_per_img_val 5 \
    --eval_img_keys_file test_img_keys.tsv \
    --cross_image_eval \
    --per_gpu_eval_batch_size 512 \
    --img_feat_file datasets/coco_ir/features.tsv \
    --pruning_steps=100 \
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
            
