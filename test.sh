python oscar/run_captioning.py \
    --do_test \
    --do_eval \
    --test_yaml test.yaml \
    --per_gpu_eval_batch_size 64 \
    --num_beams 5 \ 
    --max_gen_length 20 \
    --eval_model_dir data/output/oscar.20210415/checkpoint-59-66420/ # could be base or large models