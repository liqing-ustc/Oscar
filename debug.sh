# python oscar/run_captioning_pruning.py \
#     --model_name_or_path models/pretrained_base/checkpoint-2000000 \
#     --do_train \
#     --train_yaml val.yaml \
#     --do_lower_case \
#     --add_od_labels \
#     --learning_rate 3e-5 \
#     --per_gpu_train_batch_size 4 \
#     --num_train_epochs 60 \
#     --tie_weights \
#     --freeze_embedding \
#     --label_smoothing 0.1 \
#     --drop_worst_ratio 0.2 \
#     --drop_worst_after 20000 \
#     --output_dir output/ \
#     --self_slimming --inter_slimming --l1_loss_self_coef=1e-4 --l1_loss_inter_coef=1e-4 \
#     --pruning_strategy small --pruning_ratio 0.2 --seed 0

# python -m torch.distributed.launch --nproc_per_node 1 oscar/run_vqa_pruning.py \
#     --output_dir output/vqa \
#     --per_gpu_eval_batch_size 64 --per_gpu_train_batch_size 8 \
#     --self_slimming --inter_slimming --l1_loss_self_coef=1e-4 --l1_loss_inter_coef=1e-4 \
#     --do_train --do_lower_case --evaluate_during_training --debug 
# python -m torch.distributed.launch --nproc_per_node 1 oscar/run_vqa_pruning.py \
#     --model_name_or_path=output/vqa/Oscar_B+pruning_coef/checkpoint-0-1000 \
#     --output_dir output/vqa \
#     --per_gpu_eval_batch_size 64 --per_gpu_train_batch_size 8 \
#     --self_slimming --inter_slimming \
#     --pruning_ratio=0.2 --l1_loss_coef=1e-4 --pruning_steps=100,200,300 \
#     --do_train --do_lower_case --evaluate_during_training --debug 

# python oscar/run_pretrain_pruning.py \
# python -m torch.distributed.launch --nproc_per_node=1 oscar/run_pretrain_pruning.py \
#     --use_b 1 \
#     --max_grad_norm 10.0 --gradient_accumulation_steps 1 \
#     --use_img_layernorm 1 \
#     --output_dir output/pretrain/ \
#     --bert_model bert --model_name_or_path bert-base-uncased \
#     --do_lower_case --learning_rate 5e-05 \
#     --warmup_steps 0 --do_train --max_seq_length 35 --on_memory \
#     --max_img_seq_length 50 --img_feature_dim 2054 \
#     --drop_out 0.1 --train_batch_size 8 \
#     --ckpt_period 10000 --max_iters 2000000 --log_period 100 \
#     --data_dir data/ --dataset_file vinvl/pretrain_corpus/coco_flickr30k_googlecc_gqa_sbu_oi_x152c4big2exp168.yaml \
#     --textb_sample_mode 1 --texta_false_prob 0.25 \
#     --self_slimming \
#     --inter_slimming \
#     --pruning_ratio=0.2 \
#     --l1_loss_coef=1e-4 \
#     --pruning_steps=100,200,300

# Image-Text Retrieval
# python oscar/run_retrieval_pruning.py \
#     --model_name_or_path datasets/coco_ir/model/base/checkpoint-1340000 \
#     --do_train \
#     --do_lower_case \
#     --per_gpu_train_batch_size 16 \
#     --learning_rate 0.00002 \
#     --num_train_epochs 30 \
#     --weight_decay 0.05 \
#     --save_steps 5000 \
#     --add_od_labels \
#     --od_label_type vg \
#     --max_seq_length 70 \
#     --max_img_seq_length 70 \
#     --output_dir output/ \
#     --do_test \
#     --do_eval \
#     --test_split test \
#     --num_captions_per_img_val 5 \
#     --eval_img_keys_file test_img_keys.tsv \
#     --cross_image_eval \
#     --per_gpu_eval_batch_size 512 \
#     --img_feat_file datasets/coco_ir/features.tsv \
#     --self_slimming \
#     --inter_slimming \
#     --pruning_ratio=0.2 \
#     --l1_loss_coef=1e-4 \
#     --pruning_steps=100 \
#     --debug \

# python oscar/run_retrieval.py \
#     --do_test \
#     --do_eval \
#     --test_split test \
#     --num_captions_per_img_val 5 \
#     --eval_img_keys_file test_img_keys.tsv \
#     --cross_image_eval \
#     --per_gpu_eval_batch_size 512 \
#     --img_feat_file datasets/coco_ir/features.tsv \
#     --eval_model_dir experiments/retrieval/Oscar_B/checkpoint-29-132780 # could be base/large models.
#     # --eval_model_dir datasets/coco_ir/model/base/checkpoint-0132780 # could be base/large models.


# NLVR2
python oscar/run_nlvr.py \
    -j 4 --img_feature_dim 2054 --max_img_seq_length 40 --data_dir datasets/nlvr2 \
    --model_type bert --model_name_or_path models/pretrained_base/checkpoint-2000000 \
    --task_name nlvr --do_lower_case --max_seq_length 55 \
    --per_gpu_eval_batch_size 64 --per_gpu_train_batch_size 32 \
    --learning_rate 3e-05 --num_train_epochs 20 \
    --output_dir output --img_feature_type faster_r-cnn --data_label_type all --train_data_type all \
    --eval_data_type all --loss_type xe --save_epoch -1 --seed 88 --evaluate_during_training \
    --logging_steps -1 --drop_out 0.3 --do_train --weight_decay 0.05 --warmup_steps 10000 \
    --classifier mlp --cls_hidden_scale 3 --num_choice 2 --use_pair \
