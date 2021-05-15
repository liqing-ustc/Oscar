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


output_dir = 't-lqing/output/oscar.{}'
submit_cmd = "python -m aml_tools.aml_submit --input_dir . --output_dir {} --num_nodes 1 --exp_name t-lqing " \
             "--config_yaml ~/.azureml/{}.yaml --cmd \"{}\""
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

# job_cmd = "oscar/run_vqa.py \
#     --data_label_type mask --img_feature_type faster_r-cnn \
#     --task_name vqa_text --data_dir datasets/vqa --txt_data_dir datasets/vqa \
#     --label_file datasets/vqa/trainval_ans2label.pkl \
#     --model_type bert --model_name_or_path models/pretrained_base/checkpoint-2000000 \
#     --do_train --do_lower_case --max_seq_length 128 \
#     --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 32 \
#     --learning_rate 5e-05 --num_train_epochs 25 \
#     --save_epoch 1 --seed 88 --evaluate_during_training --logging_steps 4000 --drop_out 0.3 \
#     --weight_decay 0.05 --warmup_steps 0 --loss_type bce --img_feat_format pt \
#     --classifier linear --cls_hidden_scale 3 \
#     --output_dir output/ \
# "

job_cmd = "oscar/run_captioning_pruning.py \
    --model_name_or_path experiments/captioning/oscar.20210513.172521/checkpoint-18-20000 \
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
    --self_slimming --inter_slimming \
    --prune_before_train --self_pruning_ratio=0.333 --inter_pruning_ratio=0.4 \
"

resolved_output_dir = output_dir.format(stamp)
resolved_submit_cmd = submit_cmd.format(resolved_output_dir, ws_config, job_cmd)
print(resolved_submit_cmd)
os.system(resolved_submit_cmd)
