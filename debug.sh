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
python -m torch.distributed.launch --nproc_per_node 1 oscar/run_vqa_pruning.py \
    --model_name_or_path=output/vqa/Oscar_B+pruning_coef/checkpoint-0-1000 \
    --output_dir output/vqa \
    --per_gpu_eval_batch_size 64 --per_gpu_train_batch_size 8 \
    --self_slimming --inter_slimming \
    --pruning_ratio=0.8 --l1_loss_coef=1e-4 --pruning_steps=100 \
    --do_train --do_lower_case --evaluate_during_training --debug 