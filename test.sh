# python oscar/run_captioning.py \
#     --do_test \
#     --do_eval \
#     --test_yaml test.yaml \
#     --per_gpu_eval_batch_size 32 \
#     --num_beams 5 \
#     --max_gen_length 20 \
#     --eval_model_dir data/output/oscar.20210415/checkpoint-59-66420 # could be base or large models
#     # --eval_model_dir models/coco_captioning_base_scst/checkpoint-15-66405 # could be base or large models
#     # --eval_model_dir models/coco_captioning_base_xe/checkpoint-60-66360 # could be base or large models

python oscar/run_vqa.py \
    --do_lower_case --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 32 \
    --do_eval --output_dir=models/vqa_base