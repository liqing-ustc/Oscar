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

task = 'pretrain'
submit_cmd = "python -m aml_tools.aml_submit --input_dir . --output_dir {} --num_nodes {} --exp_name liqing-{} " \
             "--config_yaml ~/.azureml/{}.yaml --cmd \"{}\""

output_dir = 't-lqing/experiments/{}/pruning_{}_{}_{}/{}'

num_nodes = 4
train_batch_size_per_node = 1024
train_batch_size = int(train_batch_size_per_node * num_nodes)
max_iters = int(1e6 * 1024 // train_batch_size)
job_cmd = 'oscar/run_pretrain_pruning.py \
    --use_b 1 \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 \
    --use_img_layernorm 1 \
    --output_dir output/pretrain/ \
    --bert_model bert --model_name_or_path models/bert-base-uncased \
    --do_lower_case  --drop_out 0.1 \
    --warmup_steps 0 --do_train --max_seq_length 35 --on_memory \
    --max_img_seq_length 50 --img_feature_dim 2054 \
    --train_batch_size {} --learning_rate 1e-4 \
    --ckpt_period 10000 --max_iters {} --log_period 100 \
    --data_dir data/ --dataset_file vinvl/pretrain_corpus/coco_flickr30k_googlecc_gqa_sbu_oi_x152c4big2exp168.yaml \
    --textb_sample_mode 1 --texta_false_prob 0.25 \
    --self_slimming --inter_slimming --l1_loss_coef 1e-4 \
    --pruning_strategy {} --pruning_ratio {} --pruning_steps {} --seed {}\
'


n_repeat = 1
for seed in range(n_repeat):
    for pruning_steps in [2e4]:
        pruning_steps = int(pruning_steps)
        for pruning_ratio in [0.2, 0.4, 0.6, 0.8]:
            for pruning_strategy in ['small']:
                args = (pruning_strategy, pruning_ratio, pruning_steps, seed)
                resolved_output_dir = output_dir.format(task, *args)
                resolved_job_cmd = job_cmd.format(train_batch_size, max_iters, *args)
                resolved_submit_cmd = submit_cmd.format(resolved_output_dir, num_nodes, task, ws_config, resolved_job_cmd)
                print(resolved_submit_cmd)
                os.system(resolved_submit_cmd)
                # exit()
            
