#! /home/qing/.virtualenvs/azure/bin/python
import os
import pdb
from datetime import datetime

ws_config = 'itp_acv'
# ws_config = 'vlp_cust' 
# ws_config = 'vlp' 
# ws_config = 'objectdet_wu' 
stamp = ''

if not stamp:
    stamp = datetime.now().strftime('%Y%m%d.%H%M%S')

task = 'pretrain'
num_nodes = 1
output_dir = 't-lqing/output/{}/oscar.{}'.format(task, stamp)
submit_cmd = "python -m aml_tools.aml_submit --input_dir . --output_dir {} --num_nodes {} --exp_name liqing-{} " \
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
    # l1_loss_coef = 1
    # job_cmd = "oscar/run_vqa_pruning.py \
    #     --do_train --do_lower_case \
    #     --evaluate_during_training --save_steps 1000 \
    #     --self_slimming --inter_slimming --l1_loss_coef={} \
    # ".format(l1_loss_coef)
    # output_dir = 't-lqing/experiments/vqa/Oscar_B+pruning_coef_{}'.format(l1_loss_coef)

    job_cmd = "oscar/run_vqa_pruning.py \
        --do_train --do_lower_case \
        --evaluate_during_training --save_steps 100 \
        --self_slimming --inter_slimming \
        --l1_loss_coef 1e-4 --pruning_steps 10 \
        --pruning_strategy random --pruning_ratio 0.8 --seed 0\
    "

elif task == 'coco_ir':
    job_cmd = "oscar/run_retrieval.py \
    --model_name_or_path datasets/coco_ir/base/checkpoint-1340000 \
    --do_train \
    --do_lower_case \
    --evaluate_during_training \
    --num_captions_per_img_val 20 \
    --eval_caption_index_file minival_caption_indexs_top20.pt \
    --per_gpu_train_batch_size 16 \
    --learning_rate 0.00002 \
    --num_train_epochs 30 \
    --weight_decay 0.05 \
    --save_steps 5000 \
    --add_od_labels \
    --od_label_type vg \
    --max_seq_length 70 \
    --max_img_seq_length 70 \
    --output_dir output/"

elif task == 'pretrain':
    num_nodes = 2
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
        --self_slimming \
        --inter_slimming \
        --pruning_ratio=0.8 \
        --l1_loss_coef=1e-4 \
        --pruning_steps=2e4 \
    '.format(train_batch_size, max_iters)

resolved_submit_cmd = submit_cmd.format(output_dir, num_nodes, task, ws_config, job_cmd)
print(resolved_submit_cmd)
os.system(resolved_submit_cmd)
