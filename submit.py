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
output_dir = 't-lqing/output/oscar.{}'.format(stamp)
submit_cmd = "python -m aml_tools.aml_submit --input_dir . --output_dir {} --num_nodes 1 --exp_name liqing-{} " \
             "--config_yaml ~/.azureml/{}.yaml --cmd \"{}\""

if task == 'caption':
    # job_cmd = "oscar/run_captioning.py \
    #     --model_name_or_path models/pretrained_base/checkpoint-2000000 \
    #     --do_train \
    #     --do_test \
    #     --evaluate_during_training --save_steps 1000 \
    #     --do_lower_case \
    #     --add_od_labels \
    #     --learning_rate 3e-5 \
    #     --per_gpu_train_batch_size 64 \
    #     --num_train_epochs 60 \
    #     --tie_weights \
    #     --freeze_embedding \
    #     --label_smoothing 0.1 \
    #     --drop_worst_ratio 0.2 \
    #     --drop_worst_after 20000 \
    #     --output_dir output/ \
    #     --self_slimming --inter_slimming --l1_loss_self_coef=1e-4 --l1_loss_inter_coef=1e-4 \
    # "

    job_cmd = "oscar/run_captioning_pruning.py \
        --model_name_or_path experiments/captioning/Oscar_B+pruning_coef_2/checkpoint-0-1000 \
        --do_train --do_test --do_lower_case --add_od_labels --tie_weights --freeze_embedding \
        --label_smoothing 0.1 --drop_worst_ratio 0.2 --drop_worst_after 20000 --output_dir output/ \
        --evaluate_during_training --save_steps 5000 \
        --self_slimming --inter_slimming\
        --prune_before_train --self_pruning_ratio=0.666 --inter_pruning_ratio=0.8 \
        --self_pruning_method=layerwise --inter_pruning_method=global \
        --pruning_strategy random --pruning_ratio 0.2 --seed 0 \
    "

elif task == 'vqa':
    # job_cmd = "oscar/run_vqa.py \
    #     --do_train --do_lower_case \
    #     --evaluate_during_training --save_steps 1000 \
    # "
    # job_cmd = "oscar/run_vqa_pruning.py \
    #     --do_train --do_lower_case \
    #     --evaluate_during_training --save_steps 1000 \
    #     --self_slimming --inter_slimming --l1_loss_self_coef=1e-4 --l1_loss_inter_coef=1e-4\
    # "
    # output_dir = 't-lqing/experiments/vqa/Oscar_B+pruning_coef'

    job_cmd = "oscar/run_vqa_pruning.py \
        --do_train --do_lower_case \
        --model_name_or_path=experiments/vqa/Oscar_B+pruning_coef/checkpoint-0-1000 \
        --evaluate_during_training --save_steps 1000 \
        --self_slimming --inter_slimming \
        --prune_before_train --pruning_ratio=0.8 \
        --debug \
    "


resolved_submit_cmd = submit_cmd.format(output_dir, task, ws_config, job_cmd)
print(resolved_submit_cmd)
os.system(resolved_submit_cmd)
