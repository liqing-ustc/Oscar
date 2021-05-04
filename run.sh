python tools/captioning/end2end_inference.py \
    --image_file_or_path="/home/qing/Desktop/MS/responsible_ai/face_fairness_reco_imgs/ecae0967-f5b0-4b7a-9ac9-36d744b6e4bf.jpg" \
    --save_result_tsv='output/result.tsv' --save_image_tsv='output/image.tsv' --yaml='output/tmp.yaml' \
    --eval_model_dir="/home/qing/Desktop/MS/Oscar/models/coco_captioning_base_xe/checkpoint-60-66360" \
    --od_config_file=scene_graph_benchmark/sgg_configs/vgattr/vinvl_x152c4.yaml TEST.IMS_PER_BATCH 1 MODEL.WEIGHT scene_graph_benchmark/models/vinvl/vinvl_vg_x152c4.pth MODEL.ROI_HEADS.SCORE_THRESH 0.2 TEST.IGNORE_BOX_REGRESSION True MODEL.ATTRIBUTE_ON True
# python tools/captioning/end2end_inference.py \
#     --image_file_or_path="/home/qing/Desktop/MS/responsible_ai/face_fairness_reco_imgs/" \
#     --save_result_tsv='/home/qing/Desktop/MS/responsible_ai/face_fairness_reco/test.label.vinvl.nogendertag.tsv' \
#     --eval_model_dir="/home/qing/Desktop/MS/Oscar/models/coco_captioning_base_xe/checkpoint-60-66360" \
#     --od_config_file=scene_graph_benchmark/sgg_configs/vgattr/vinvl_x152c4.yaml TEST.IMS_PER_BATCH 1 MODEL.WEIGHT scene_graph_benchmark/models/vinvl/vinvl_vg_x152c4.pth MODEL.ROI_HEADS.SCORE_THRESH 0.2 TEST.IGNORE_BOX_REGRESSION True MODEL.ATTRIBUTE_ON True