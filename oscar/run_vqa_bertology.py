#!/usr/bin/env python3
# Copyright 2018 CMU and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Bertology: this script shows how you can explore the internals of the models in the library to:
    - compute the entropy of the head attentions
    - compute the importance of each head
    - prune (remove) the low importance head.
    Some parts of this script are adapted from the code of Michel et al. (http://arxiv.org/abs/1905.10650)
    which is available at https://github.com/pmichel31415/are-16-heads-really-better-than-1
"""
from datetime import datetime
import argparse
import base64
import numpy as np
import os
import os.path as op
import random, time, json
import torch
import torch.distributed as dist
from torch.utils import data
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from run_vqa import *

logger = logging.getLogger(__name__)

def entropy(p):
    """Compute the entropy of a probability distribution"""
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)


def print_2d_tensor(tensor):
    """Print a 2D tensor"""
    logger.info("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
    for row in range(len(tensor)):
        if tensor.dtype != torch.long:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
        else:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))


def compute_heads_importance(
    args, model, eval_dataloader, compute_entropy=True, compute_importance=True, head_mask=None, actually_pruned=False
):
    """This method shows how to compute:
    - head attention entropy
    - head importance scores according to http://arxiv.org/abs/1905.10650
    """
    # Prepare our tensors
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(args.device)
    attn_entropy = torch.zeros(n_layers, n_heads).to(args.device)
    attn_kl = torch.zeros(n_layers*n_heads, n_layers*n_heads).to(args.device)

    if head_mask is None:
        head_mask = torch.ones(n_layers, n_heads).to(args.device)

    head_mask.requires_grad_(requires_grad=True)
    # If actually pruned attention multi-head, set head mask to None to avoid shape mismatch
    if actually_pruned:
        head_mask = None

    preds = None
    labels = None
    tot_tokens = 0.0

    # model.train()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
        batch = tuple(t.to(args.device) for t in batch)

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        inputs = {'input_ids':      batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                'labels':         batch[4],
                'img_feats':      None if args.img_feature_dim == -1 else batch[5]}
        outputs = model(**inputs, head_mask=head_mask)
        loss, logits, all_attentions = (
            outputs[0],
            outputs[1],
            outputs[-1],
        )  # Loss and logits are the first, attention the last

        if compute_entropy:
            for layer, attn in enumerate(all_attentions):
                masked_entropy = entropy(attn.detach()) * inputs["attention_mask"].float().unsqueeze(1)
                attn_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()
        
        if True:
            m = torch.stack(all_attentions, axis=1).detach()
            m = m.reshape((m.shape[0], -1, m.shape[3], m.shape[4]))
            m = torch.einsum('biqk,bjqk->bqij', m, (m+1e-10).log())
            m = m.diagonal(dim1=-2, dim2=-1).unsqueeze(-1) - m
            m = torch.einsum('bqij,bq->ij', m, inputs['attention_mask'].float())
            m = (m + m.T) / 2
            attn_kl += m

        if compute_importance:
            head_mask.grad = None
            loss.backward()  # Backpropagate to populate the gradients in the head mask
            head_importance += head_mask.grad.abs().detach()

        # Also store our logits/labels if we want to compute metrics afterwards
        if preds is None:
            preds = logits.detach().cpu()
            labels = inputs["labels"].detach().cpu()
        else:
            preds = torch.cat((preds, logits.detach().cpu()), 0)
            labels = torch.cat((labels, inputs["labels"].detach().cpu()), 0)

        tot_tokens += inputs["attention_mask"].float().detach().cpu().sum().data


    # Normalize
    attn_entropy /= tot_tokens
    head_importance /= tot_tokens

    if True:
        attn_kl /= tot_tokens
        np.save('output/attn_kl.npy', attn_kl.cpu().numpy())
        import matplotlib.pyplot as plt
        plt.imshow(attn_kl.cpu())
        plt.colorbar()
        plt.savefig("output/attn_kl.png", dpi=1000)
        plt.show()

    # Layerwise importance normalization
    if not args.dont_normalize_importance_by_layer:
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

    if not args.dont_normalize_global_importance:
        head_importance = (head_importance - head_importance.min()) / (head_importance.max() - head_importance.min())


    head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=args.device)
    head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
        head_importance.numel(), device=args.device
    )
    head_ranks = head_ranks.view_as(head_importance)
    
    # Print/save matrices
    # np.save(os.path.join(args.output_dir, "attn_entropy.npy"), attn_entropy.detach().cpu().numpy())
    # np.save(os.path.join(args.output_dir, "head_importance.npy"), head_importance.detach().cpu().numpy())
    # if compute_entropy:
    #     logger.info("Attention entropies")
    #     print_2d_tensor(attn_entropy)
    # if compute_importance:
    #     logger.info("Head importance scores")
    #     print_2d_tensor(head_importance)
    #     logger.info("Head ranked by importance scores")
    #     print_2d_tensor(head_ranks)

    return attn_entropy, head_importance, preds, labels


def mask_heads(args, model, eval_dataloader):
    """This method shows how to mask head (set some heads to zero), to test the effect on the network,
    based on the head importance scores, as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    _, head_importance, preds, labels = compute_heads_importance(args, model, eval_dataloader, compute_entropy=False)
    # preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
    # original_score = glue_compute_metrics(args.task_name, preds, labels)[args.metric_name]
    original_score = torch.sum(compute_score_with_logits(preds, labels), 1).mean()

    logger.info("Pruning: original score: %f, threshold: %f", original_score, original_score * args.masking_threshold)

    new_head_mask = torch.ones_like(head_importance)
    num_to_mask = max(1, int(new_head_mask.numel() * args.masking_amount))

    current_score = original_score
    while current_score >= original_score * args.masking_threshold:
        head_mask = new_head_mask.clone()  # save current head mask
        # heads from least important to most - keep only not-masked heads
        head_importance[head_mask == 0.0] = float("Inf")
        current_heads_to_mask = head_importance.view(-1).sort()[1]

        if len(current_heads_to_mask) <= num_to_mask:
            break

        # mask heads
        current_heads_to_mask = current_heads_to_mask[:num_to_mask]
        logger.info("Heads to mask: %s", str(current_heads_to_mask.tolist()))
        new_head_mask = new_head_mask.view(-1).clone()
        new_head_mask[current_heads_to_mask] = 0.0
        new_head_mask = new_head_mask.view_as(head_mask)
        new_head_mask = new_head_mask.clone().detach()
        # print_2d_tensor(new_head_mask)

        # Compute metric and head importance again
        _, head_importance, preds, labels = compute_heads_importance(
            args, model, eval_dataloader, compute_entropy=False, head_mask=new_head_mask
        )
        # preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
        # current_score = glue_compute_metrics(args.task_name, preds, labels)[args.metric_name]
        current_score = torch.sum(compute_score_with_logits(preds, labels), 1).mean()
        logger.info(
            "Masking: current score: %f, remaining heads %d (%.1f percents)",
            current_score,
            new_head_mask.sum(),
            new_head_mask.sum() / new_head_mask.numel() * 100,
        )

    logger.info("Final head mask")
    print_2d_tensor(head_mask)
    print((head_mask == 0).nonzero())
    # np.save(os.path.join(args.output_dir, "head_mask.npy"), head_mask.detach().cpu().numpy())

    return head_mask


def prune_heads(args, model, eval_dataloader, head_mask):
    """This method shows how to prune head (remove heads weights) based on
    the head importance scores as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    # Try pruning and test time speedup
    # Pruning is like masking but we actually remove the masked weights
    before_time = datetime.now()
    _, _, preds, labels = compute_heads_importance(
        args, model, eval_dataloader, compute_entropy=False, compute_importance=False, head_mask=head_mask
    )
    # preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
    # score_masking = glue_compute_metrics(args.task_name, preds, labels)[args.metric_name]
    score_masking = torch.sum(compute_score_with_logits(preds, labels), 1).mean()
    original_time = datetime.now() - before_time

    original_num_params = sum(p.numel() for p in model.parameters())
    heads_to_prune = dict(
        (layer, (1 - head_mask[layer].long()).nonzero().squeeze(1).tolist()) for layer in range(len(head_mask))
    )

    assert sum(len(h) for h in heads_to_prune.values()) == (1 - head_mask.long()).sum().item()
    model.prune_heads(heads_to_prune)
    pruned_num_params = sum(p.numel() for p in model.parameters())

    before_time = datetime.now()
    _, _, preds, labels = compute_heads_importance(
        args,
        model,
        eval_dataloader,
        compute_entropy=False,
        compute_importance=False,
        head_mask=None,
        actually_pruned=True,
    )
    # preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
    # score_pruning = glue_compute_metrics(args.task_name, preds, labels)[args.metric_name]
    score_pruning = torch.sum(compute_score_with_logits(preds, labels), 1).mean()
    new_time = datetime.now() - before_time

    logger.info(
        "Pruning: original num of params: %.2e, after pruning %.2e (%.1f percents)",
        original_num_params,
        pruned_num_params,
        pruned_num_params / original_num_params * 100,
    )
    logger.info("Pruning: score with masking: %f score with pruning: %f", score_masking, score_pruning)
    logger.info("Pruning: speed ratio (new timing / original timing): %f percents", original_time / new_time * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="datasets/vqa", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--txt_data_dir", default="datasets/vqa", type=str,
                        help="The input text data dir. Should contain the .json files (or other data files) for the task.")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default="models/pretrained_base/checkpoint-2000000", type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--task_name", default="vqa_text", type=str,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--label_file", type=str, default="datasets/vqa/trainval_ans2label.pkl", help="Label Dictionary")
    parser.add_argument("--label2ans_file", type=str, default="datasets/vqa/trainval_label2ans.pkl", help="Label to Answer Dictionary")

    parser.add_argument("--img_feat_dir", default=None, type=str, help="The input img_feat_dir.")
    parser.add_argument("--img_feat_format", default='pt', type=str, help="img_feat_format: pt or tsv.")

    parser.add_argument("--data_label_type", default='mask', type=str, help="faster or mask")
    parser.add_argument("--loss_type", default='bce', type=str, help="kl or xe")
    parser.add_argument("--use_vg", action='store_true', help="Use VG-QA or not.")
    parser.add_argument("--use_vg_dev", action='store_true', help="Use VG-QA as validation.")
    #parser.add_argument("--use_img_layernorm", action='store_true', help="use_img_layernorm")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train_val", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run test on the test set.")
    parser.add_argument("--do_test_dev", action='store_true', help="Whether to run test on the test-dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true', help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

    parser.add_argument("--drop_out", default=0.3, type=float, help="Drop out for BERT.")
    parser.add_argument("--adjust_dp",action='store_true', help="Adjust Drop out for BERT.")

    parser.add_argument("--adjust_loss", action='store_true', help="Adjust Loss Type for BERT.")
    parser.add_argument("--adjust_loss_epoch", default=-1, type=int, help="Adjust Loss Type for BERT.")
    parser.add_argument("--classifier", default='linear', type=str, help="linear or mlp")
    parser.add_argument("--cls_hidden_scale", default=3, type=int, help="cls_hidden_scale: for classifier")

    parser.add_argument("--hard_label", action='store_true', help="Soft Label or Hard Label.")

    parser.add_argument("--max_img_seq_length", default=50, type=int, help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int, help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='faster_r-cnn', type=str, help="faster_r-cnn or mask_r-cnn")
    parser.add_argument("--code_voc", default=512, type=int, help="dis_code_voc: 256, 512")
    parser.add_argument("--code_level", default='top', type=str, help="code level: top, botttom, both")

    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=256, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=25, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=4000, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=-1, help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_epoch', type=int, default=1, help="Save checkpoint every X epochs.")
    parser.add_argument('--save_after_epoch', type=int, default=-1, help="Save checkpoint after epoch.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true', help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true', help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    parser.add_argument("--philly", action='store_true', help="Use Philly: reset the output dir")
    parser.add_argument("--load_fast", action='store_true', help="Load Tensor Fast")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

    # for pruning
    parser.add_argument(
        "--data_subset", type=int, default=-1, help="If > 0: limit the data to a subset of data_subset instances."
    )

    parser.add_argument(
        "--dont_normalize_importance_by_layer", action="store_true", help="Don't normalize importance score by layers"
    )
    parser.add_argument(
        "--dont_normalize_global_importance",
        action="store_true",
        help="Don't normalize all importance scores between 0 and 1",
    )

    parser.add_argument(
        "--try_masking", action="store_true", help="Whether to try to mask head until a threshold of accuracy."
    )
    parser.add_argument(
        "--masking_threshold",
        default=0.9,
        type=float,
        help="masking threshold in term of metrics (stop masking when metric < threshold * original metric value).",
    )
    parser.add_argument(
        "--masking_amount", default=0.1, type=float, help="Amount to heads to masking at each masking step."
    )
    parser.add_argument("--metric_name", default="acc", type=str, help="Metric to use for head masking.")
    args = parser.parse_args()


    if args.philly:  # use philly
        logger.info('Info: Use Philly, all the output folders are reset.')
        args.output_dir = os.path.join(os.getenv('PT_OUTPUT_DIR'), args.output_dir)
        logger.info('OUTPUT_DIR:', args.output_dir)

    #if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    #    raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train: logger.info("Output Directory Exists.")

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        logger.info("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed, args.n_gpu)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))

    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels(args.label_file)
    num_labels = len(label_list)
    logger.info('Task Name: {}, #Labels: {}'.format(args.task_name, num_labels))

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels, finetuning_task=args.task_name,
    )
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

    # discrete code
    config.img_feature_dim = args.img_feature_dim
    config.img_feature_type = args.img_feature_type
    config.code_voc = args.code_voc
    config.hidden_dropout_prob = args.drop_out
    config.loss_type = args.loss_type
    config.classifier = args.classifier
    config.cls_hidden_scale = args.cls_hidden_scale
    #config.use_img_layernorm = args.use_img_layernorm
    
    # load discrete code
    if args.img_feature_type in ['dis_code', 'dis_code_t']:
        logger.info('Load discrete code from: {}'.format(args.data_dir))
        t_start = time.time()
        train_code = torch.load(os.path.join(args.data_dir, 'vqvae', 'train.pt'))
        t_end = time.time()
        logger.info('Load time: %.3f' % (t_end - t_start))

        if args.code_level == 'top':
            config.code_dim = train_code['embeddings_t'].shape[0]
            config.code_size = train_code['feats_top'][list(train_code['feats_top'].keys())[0]].shape[0]
        elif args.code_level == 'bottom':
            config.code_dim = train_code['embeddings_b'].shape[0]
            config.code_size = train_code['feats_bottom'][list(train_code['feats_bottom'].keys())[0]].shape[0]
        elif args.code_level == 'both':
            config.code_dim = train_code['embeddings_t'].shape[0] + train_code['embeddings_b'].shape[0]

    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    if args.img_feature_type in ['dis_code', 'dis_code_t']:
        logger.info('Initializing the code embedding with {}'.format(args.code_level))
        if args.code_level == 'top':
            model.init_code_embedding(train_code['embeddings_t'].t())
        elif args.code_level == 'bottom':
            model.init_code_embedding(train_code['embeddings_b'].t())

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    #if args.do_eval:
    eval_dataset = VQADataset(args, 'val', tokenizer)

    if args.do_test:
        test_dataset = VQADataset(args, 'test2015', tokenizer)

    if args.do_test_dev:
        test_dev_dataset = VQADataset(args, 'test-dev2015', tokenizer)


    checkpoint = args.output_dir
    config.output_attentions = True
    model = model_class.from_pretrained(checkpoint, config=config)
    model.to(args.device)
    # result, score, upper_bound = evaluate(args, model, eval_dataset)
    
    # Compute head entropy and importance score
    if args.data_subset > 0:
        dataset = Subset(eval_dataset, list(range(min(args.data_subset, len(eval_dataset)))))
    else:
        dataset = eval_dataset
    eval_dataloader = DataLoader(dataset, num_workers=args.workers, batch_size=args.per_gpu_train_batch_size)
    # compute_heads_importance(args, model, eval_dataloader)

    # Try head masking (set heads to zero until the score goes under a threshole)
    # and head pruning (remove masked heads and see the effect on the network)
    if args.try_masking and args.masking_threshold > 0.0 and args.masking_threshold < 1.0:
        head_mask = mask_heads(args, model, eval_dataloader)
        prune_heads(args, model, eval_dataloader, head_mask)
        result, score, upper_bound = evaluate(args, model, eval_dataset)



if __name__ == "__main__":
    main()
