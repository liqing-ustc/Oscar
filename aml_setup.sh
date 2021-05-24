pip install -r requirements.txt
ln -s $AML_JOB_INPUT_PATH/t-lqing/datasets
ln -s $AML_JOB_INPUT_PATH/t-lqing/models
ln -s $AML_JOB_INPUT_PATH/t-lqing/coco_caption
ln -s $AML_JOB_INPUT_PATH/t-lqing/experiments
ln -s $AML_JOB_INPUT_PATH/t-lqing data
ln -s $AML_JOB_OUTPUT_PATH output
df -h
ls -al
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_DEBUG=INFO
