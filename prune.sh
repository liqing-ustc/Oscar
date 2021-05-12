# python oscar/run_pruning.py \
#     --do_test \
#     --do_eval \
#     --test_yaml test.yaml \
#     --per_gpu_train_batch_size 32 \
#     --num_beams 1 \
#     --max_gen_length 20 \
#     --try_masking --masking_threshold=0.95 \
#     --eval_model_dir=models/coco_captioning_base_xe/checkpoint-60-66360 # could be base or large models
#     # --data_subset=1000 \
#     # --eval_model_dir models/coco_captioning_base_scst/checkpoint-15-66405 # could be base or large models
#     # --eval_model_dir models/coco_captioning_base_xe/checkpoint-60-66360 # could be base or large models

python oscar/run_vqa_pruning.py \
    --per_gpu_eval_batch_size=128 --per_gpu_train_batch_size=16 \
    --do_train --do_lower_case --output_dir=models/vqa_base \
    --try_masking --masking_threshold=0.95