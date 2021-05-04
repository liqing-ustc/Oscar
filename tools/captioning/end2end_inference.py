import sys
import argparse
import os
import os.path as op
import time
import json
import torch
from tqdm import tqdm


sys.path.append("./scene_graph_benchmark")
from maskrcnn_benchmark.config import cfg
from scene_graph_benchmark.config import sg_cfg
from captioning_e2e import BertCaptioning, ObjectDetector, convert_prediction_results

def restore_training_settings(args):
    # restore training settings
    train_args = torch.load(op.join(args.eval_model_dir, 'training_args.bin'))
    assert hasattr(train_args, 'max_seq_a_length')
    max_od_labels_len = train_args.max_seq_length - train_args.max_seq_a_length
    max_seq_length = args.max_gen_length + max_od_labels_len
    args.max_seq_length = max_seq_length
    print('Override max_seq_length to {} = max_gen_length:{} + od_labels_len:{}'.format(
            max_seq_length, args.max_gen_length, max_od_labels_len))

    override_params = [
            'od_label_conf', 'max_seq_a_length',
            'do_lower_case', 'add_od_labels',
            'max_img_seq_length', 'img_feature_dim']
    for param in override_params:
        assert hasattr(train_args, param)
        train_v = getattr(train_args, param)
        test_v = getattr(args, param)
        if train_v != test_v:
            print('Override {} with train args: {} -> {}'.format(param, test_v, train_v))
            setattr(args, param, train_v)

    force_params = {
        # 'unique_labels_on': True,
        'no_sort_by_conf': False,
    }
    for param in force_params:
        assert hasattr(train_args, param)
        train_v = getattr(train_args, param)
        if train_v != force_params[param]:
            raise ValueError('Model parameter not supported {}: {}'.format(param, train_v))
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_file_or_path", default='datasets/inference_test/', type=str, required=False,
                        help="single image or a folder with multiple images, or a yaml file for inference")
    parser.add_argument("--eval_model_dir", default='', type=str, required=True,
                        help="model directory with checkpoint for inference")
    parser.add_argument("--od_config_file", default='', type=str, required=True,
                        help="config file to build OD model")
    parser.add_argument("--save_result_tsv", default='', type=str, required=False,
                        help='file name to save results, skip saving if not given')
    parser.add_argument("--save_image_tsv", default='', type=str, required=False,
                        help='file name to save results, skip saving if not given')
    parser.add_argument("--yaml", default='', type=str, required=False,
                        help='file name to save yaml, skip saving if not given')
    parser.add_argument("--save_models_path", default='', type=str, required=False,
                        help='file path to save models, skip saving if not given')
    parser.add_argument("--max_seq_length", default=70, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--max_seq_a_length", default=40, type=int, 
                        help="The maximum sequence length for caption.")
    parser.add_argument("--do_lower_case", action='store_true', 
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--add_od_labels", default=False, action='store_true', 
                        help="Whether to add object detection labels or not")
    parser.add_argument("--max_img_seq_length", default=50, type=int, 
                        help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int, 
                        help="The Image Feature Dimension.")
    parser.add_argument("--od_label_conf", default=0.0, type=float, 
                        help="Confidence threshold to select od labels.")
    parser.add_argument("--load_tsv_path", default='', type=str, required=False, 
                        help="For debugging only, load OD features and labels"
                        "from the folder path")
    parser.add_argument("--device", default='cpu', type=str, required=False, 
                        help="Inference device: cuda or cpu")
    parser.add_argument("--cpu_threads", default=0, type=int, required=False,
                        help="number of cpu threads (0-default means not limiting threads)")
    parser.add_argument("--num_try", default=1, type=int, required=False, 
                        help="Number of trials to average inference speed.")
    parser.add_argument("--forward_time_checker", action='store_true',
                        help="Use time checker from quickdetection.")
    # for caption generation
    parser.add_argument('--mode', type=str, default="default",
                        help="Choose from prod, prod_no_hidden for simplified "
                        "generate method, with output_hidden_states set to "
                        "True, False, respectively. The default uses origianl "
                        "generate method with more functionalities")
    parser.add_argument('--max_gen_length', type=int, default=20,
                        help="max length of generated sentences")
    parser.add_argument('--output_hidden_states', action='store_true',
                        help="Turn on for fast decoding")
    parser.add_argument('--num_return_sequences', type=int, default=1,
                        help="repeating times per image")
    parser.add_argument('--num_beams', type=int, default=1, 
                        help="beam search width")
    parser.add_argument('--num_keep_best', type=int, default=1,
                        help="number of hypotheses to keep in beam search")
    parser.add_argument('--temperature', type=float, default=1,
                        help="temperature in softmax for sampling")
    parser.add_argument('--top_k', type=int, default=0,
                        help="filter distribution for sampling")
    parser.add_argument('--top_p', type=float, default=1,
                        help="filter distribution for sampling")
    parser.add_argument('--repetition_penalty', type=int, default=1,
                        help="repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)")
    parser.add_argument('--length_penalty', type=int, default=1,
                        help="beam search length penalty")
    # for object detection additional arguments to modify config
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()   
    args = restore_training_settings(args) # for captioning
    
    bert_device = torch.device(args.device)
    cpu_device = torch.device('cpu')

    print("Build object detection model.")
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.od_config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)

    detector = ObjectDetector(cfg)
    detector.model.to(cfg.MODEL.DEVICE)
    detector.model.eval()
    if args.forward_time_checker:
        from qd.layers import ForwardPassTimeChecker
        detector.model = ForwardPassTimeChecker(detector.model)

    print("Build BERT model.")
    print("Inference mode: {}".format(args.mode))
    captioner = BertCaptioning(args)
    total_params = sum(p.numel() for p in captioner.model.parameters())
    print('Total Parameters: {}'.format(total_params))
    captioner.model.to(bert_device)
    captioner.model.eval()

    if args.save_models_path:
        print("Saving models.")
        torch.save({"detector": detector, "captioner": captioner, "args": vars(args)}, args.save_models_path)
        return
    if not args.image_file_or_path:
        raise ValueError("image_file_or_path not specified")
    if op.isfile(args.image_file_or_path):
        image_list = [args.image_file_or_path]
    elif op.isdir(args.image_file_or_path):
        # should be a path with images only
        image_list = sorted(os.listdir(args.image_file_or_path))
        image_list = [op.join(args.image_file_or_path, img) for img in image_list]
    else:
        raise ValueError("Cannot find images at {}".format(args.image_file_or_path))

    print("Start inference")
    od_times = []
    bert_times = []
    results = []
    for image_file in tqdm(image_list):
        od_time = []
        try: 
            for i in range(args.num_try):
                tic = time.time()
                image, scale = detector.pre_processing(image_file)
                image.to(cfg.MODEL.DEVICE)
                predictions = detector(image, scale)
                od_time.append(time.time() - tic)
                # print("OD inference time: {}".format(od_time[-1]))
            od_times.append(sum(od_time) / args.num_try)
        except AssertionError:
            print("invalid image: ", image_file)

        predictions = predictions[0].to(cpu_device)
        bert_time = []
        for i in range(args.num_try):
            tic = time.time()
            img_feats, od_labels = convert_prediction_results(predictions, image, 
                detector.labelmap_invert, od_label_conf=args.od_label_conf,
                load_tsv_path=args.load_tsv_path)
            od_labels = [x if x not in ['man', 'woman', 'boy', 'girl', 'little girl', 'young man', 'gentleman'] else 'person' for x in od_labels]
            batch = captioner.pre_processing(img_feats, od_labels)
            batch = tuple([torch.unsqueeze(t, 0).to(bert_device) for t in batch])
            result = captioner(*batch)
            bert_time.append(time.time() - tic)
            # print("BERT inference time: {}".format(bert_time[-1]))
            if i == 0:
                img_key = op.basename(image_file).replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
                results.append([img_key, json.dumps(result)])
                # print(result, op.basename(image_file))
        bert_times.append(sum(bert_time) / args.num_try)
    
    if args.forward_time_checker:
        from maskrcnn_benchmark.utils.miscellaneous import write_to_yaml_file
        x = detector.model.get_time_info()
        write_to_yaml_file(x, 'speed.yaml')

    print("Average OD inference time: {}".format(sum(od_times) / len(image_list)))
    print("Average BERT inference time: {}".format(sum(bert_times) / len(image_list)))

    if args.save_result_tsv:
        from maskrcnn_benchmark.structures.tsv_file_ops import tsv_writer
        print('save results to: ', args.save_result_tsv)
        tsv_writer(results, args.save_result_tsv)
    if args.save_image_tsv:
        import cv2, base64
        images = []
        for image_file in image_list:
            img = cv2.imread(image_file)
            img_encoded_str = base64.b64encode(cv2.imencode('.jpg', img)[1])
            images.append([op.basename(image_file), img_encoded_str])
        tsv_writer(images, args.save_image_tsv)
        
        with open(args.yaml, 'w') as f:
            f.write("img: " + op.basename(args.save_image_tsv) + "\n")
            f.write("label: " + op.basename(args.save_result_tsv) + "\n")




if __name__ == "__main__":
    main()