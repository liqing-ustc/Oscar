# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import os.path as op
import copy, time, json
import base64

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import _pickle as cPickle

from oscar.modeling.modeling_bert import ImageBertForSequenceClassification
from transformers.pytorch_transformers import WEIGHTS_NAME, BertTokenizer, BertConfig
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule

from oscar.utils.logger import setup_logger
from oscar.utils.misc import set_seed, mkdir, save_checkpoint, synchronize, prepare_model_optimizer
from oscar.utils.task_utils import (_truncate_seq_pair, convert_examples_to_features_vqa,
                        output_modes, processors)
from oscar.utils.pruning import prune, count_flops, calculate_l1_loss
from tqdm import tqdm

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, ImageBertForSequenceClassification, BertTokenizer),
}


log_json = []
debug_size = 500


def _load_dataset(args, name):
    processor = processors[args.task_name]()
    labels = processor.get_labels(args.label_file)

    if name == 'train':
        if args.data_label_type == 'mask':
            if args.use_vg:
                #examples = processor.get_train_examples(args.data_dir, 'train2014_vg_qla_mrcnn.json')
                examples = processor.get_train_examples(args.txt_data_dir, 'train2014_vg_qla_mrcnn.json')
            else:
                examples = processor.get_train_examples(args.txt_data_dir, 'train2014_qla_mrcnn.json')
        else:
            examples = processor.get_train_examples(args.txt_data_dir, 'train2014_qla.json')
    elif name == 'val':
        if args.data_label_type == 'mask':
            if args.use_vg_dev:
                examples = processor.get_dev_examples(args.txt_data_dir, 'vg_qla_mrcnn.json')
            else:
                examples = processor.get_dev_examples(args.txt_data_dir, 'val2014_qla_mrcnn.json')
        else:
            examples = processor.get_dev_examples(args.txt_data_dir, 'val2014_qla.json')
    elif name == 'train+val':
        if args.data_label_type == 'mask':
            examples = processor.get_train_examples(args.txt_data_dir, 'train+val2014_qla_mrcnn.json')
            #examples = processor.get_train_examples(args.data_dir, 'train+val2014_qla_mrcnn.json')
        else:
            examples = processor.get_train_examples(args.txt_data_dir, 'train+val2014_qla.json')
    elif name == 'test2015':
        if args.data_label_type == 'mask':
            examples = processor.get_test_examples(args.data_dir, 'test2015_qla_mrcnn.json')
        else:
            examples = processor.get_test_examples(args.data_dir, 'test2014_qla.json')
    elif name == 'test-dev2015':
        if args.data_label_type == 'mask':
            examples = processor.get_test_examples(args.data_dir, 'test-dev2015_qla_mrcnn.json')
        else:
            examples = processor.get_test_examples(args.data_dir, 'test2014_qla.json')

    return examples, labels


class VQADataset(Dataset):
    """ VQA Dataset """

    def __init__(self, args, name, tokenizer):
        super(VQADataset, self).__init__()
        assert name in ['train', 'val', 'test-dev2015', 'test2015', 'train+val']

        self.args = args
        self.name = name

        # load image features
        t_start = time.time()
        self.img_feature_file = None
        self.img_feat_offset_map = None

        if args.img_feature_type == 'faster_r-cnn':
            if args.img_feat_format == 'pt':
                if args.img_feature_dim == 2048: # object features
                    self.img_features = torch.load(os.path.join(args.data_dir, '{}_img_frcnn_obj_feats.pt'.format(name)))
                else: # object + spatial features
                    if args.use_vg_dev:
                        self.img_features = torch.load(os.path.join(args.data_dir, 'train+val_img_frcnn_feats.pt'))
                    else:
                        self.img_features = torch.load(os.path.join(args.data_dir, '{}_img_frcnn_feats.pt'.format(name)))
            elif args.img_feat_format == 'tsv':
                self.load_img_tsv_features()
        elif args.img_feature_type == 'mask_r-cnn':
            self.img_features = torch.load(os.path.join(args.data_dir, '{}_img_mask_rcnn_feats.pt'.format(name)))
        elif args.img_feature_type.startswith('dis_code'): #in ['dis_code', 'dis_code_t']: # discrete code
            self.img_features = torch.load(os.path.join(args.data_dir, 'vqvae', '{}.pt'.format(name)))['feats_{}'.format(args.code_level)]
        else:
            self.img_features = torch.load(os.path.join(args.data_dir, '{}_img_feats.pt'.format(name)))
        t_end = time.time()
        logger.info('Info: loading {0} features using {1:.2f} secs'.format(name, (t_end-t_start)))

        self.output_mode = output_modes[args.task_name]
        self.tokenizer = tokenizer

        self.examples, self.labels = _load_dataset(args, name)
        self.label_map = {label: i for i, label in enumerate(self.labels)}

        if self.args.load_fast:
            self.features = self.tensorize(args, cls_token_at_end=bool(self.args.model_type in ['xlnet']), # xlnet has a cls token at the end
                cls_token=self.tokenizer.cls_token,
                sep_token=self.tokenizer.sep_token,
                cls_token_segment_id=2 if self.args.model_type in ['xlnet'] else 0,
                pad_on_left=bool(self.args.model_type in ['xlnet']), # pad on the left for xlnet
                pad_token_segment_id=4 if self.args.model_type in ['xlnet'] else 0)
        else:
            pass

        logger.info('%s Data Examples: %d' % (name, len(self.examples)))


    def tensorize(self, cls_token_at_end=False, pad_on_left=False,
                    cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                    sequence_a_segment_id=0, sequence_b_segment_id=1,
                    cls_token_segment_id=1, pad_token_segment_id=0,
                    mask_padding_with_zero=True):

        # debug:
        debug_size = 500
        features = []

        for (ex_index, example) in enumerate(self.examples[0: ]):
            if len(example.label) == 0: continue
            if ex_index % 10000 == 0: logger.info("Tensorizing example %d of %d" % (ex_index, len(self.examples)))

            tokens_a = self.tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = self.tokenizer.tokenize(example.text_b)
                # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > self.args.max_seq_length - 2:
                    tokens_a = tokens_a[:(self.args.max_seq_length - 2)]

            tokens = tokens_a + [sep_token]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if tokens_b:
                tokens += tokens_b + [sep_token]
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = self.args.max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == self.args.max_seq_length
            assert len(input_mask) == self.args.max_seq_length
            assert len(segment_ids) == self.args.max_seq_length

            # image features
            img_feat = self.img_features[example.img_key] # torch
            #img_feat = self.img_features.item().get(example.img_key)  # numpy
            if img_feat.shape[0] > self.args.max_img_seq_length:
                img_feat = img_feat[0:self.args.max_img_seq_length, ]
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                    # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
            else:
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                    # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]
                padding_matrix = torch.zeros((self.args.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
                img_feat = torch.cat((img_feat, padding_matrix), 0)
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])
                    # segment_ids = segment_ids + [pad_token_segment_id] * padding_matrix.shape[0]

            if self.args.output_mode == "classification":
                label_id = [self.label_map[l] for l in example.label]
                score = example.score
            elif self.args.output_mode == "regression":
                label_id = float(example.label)
            else:
                raise KeyError(self.args.output_mode)

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label: %s (id = %s)" % (example.label, label_id))
                logger.info("score: %s (score = %s)" % (example.score, score))

            new_scores = target_tensor(len(self.labels), label_id, score)
            #features.append(InputFeat(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id, score=score, img_feat=img_feat))
            features.append((torch.tensor(input_ids, dtype=torch.long),
                            torch.tensor(input_mask, dtype=torch.long),
                            torch.tensor(segment_ids, dtype=torch.long),
                            torch.tensor([label_id[0]], dtype=torch.long),
                            torch.tensor(new_scores, dtype=torch.float), img_feat))

        return features

    def tensorize_example(self, example, cls_token_at_end=False, pad_on_left=False,
                    cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                    sequence_a_segment_id=0, sequence_b_segment_id=1,
                    cls_token_segment_id=1, pad_token_segment_id=0,
                    mask_padding_with_zero=True):

        tokens_a = self.tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = self.tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self.args.max_seq_length - 2:
                tokens_a = tokens_a[:(self.args.max_seq_length - 2)]

        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.args.max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == self.args.max_seq_length
        assert len(input_mask) == self.args.max_seq_length
        assert len(segment_ids) == self.args.max_seq_length

        # image features
        if self.args.img_feature_type.startswith('dis_code'):
            img_feat = self.img_features[example.img_key]

            if self.args.img_feature_type == 'dis_code_ln': # for discrete code image representation
                img_feat = img_feat.reshape(-1, img_feat.shape[0])

            if self.args.img_feature_type == 'dis_code_t': # transposed
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * 64
            else:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
        else:
            if self.args.img_feat_format == 'pt':
                img_feat = self.img_features[example.img_key] #[:, 0:self.args.img_feature_dim]  # torch
            elif self.args.img_feat_format == 'tsv':
                img_features = self.get_img_feature(str(example.img_key))
                img_feat = torch.from_numpy(img_features)

            if img_feat.shape[0] > self.args.max_img_seq_length:
                img_feat = img_feat[0:self.args.max_img_seq_length, ]
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                    # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
            else:
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                    # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]
                padding_matrix = torch.zeros((self.args.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
                img_feat = torch.cat((img_feat, padding_matrix), 0)
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])
                    # segment_ids = segment_ids + [pad_token_segment_id] * padding_matrix.shape[0]

        if self.args.output_mode == "classification":
            if (example.label is None):
                label_id = [0]
                score = [0]
            elif len(example.label) == 0:
                label_id = [0]
                score = [0]
            else:
                label_id = [self.label_map[l] for l in example.label]
                score = example.score
        elif self.args.output_mode == "regression":
            if len(example.label) == 0:
                label_id = 0
            else:
                label_id = float(example.label)
        else:
            raise KeyError(self.args.output_mode)

        new_scores = target_tensor(len(self.labels), label_id, score)

        # features.append(InputFeat(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id, score=score, img_feat=img_feat))
        if self.args.img_feature_type in ['dis_code', 'dis_code_t']:
            img_feat = img_feat.type(torch.long)
        elif self.args.img_feature_type in ['dis_code_ln']:
            #img_feat = img_feat.reshape(-1, img_feat.shape[0])
            img_feat = img_feat.type(torch.float)

        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(input_mask, dtype=torch.long),
                torch.tensor(segment_ids, dtype=torch.long),
                torch.tensor([label_id[0]], dtype=torch.long),
                torch.tensor(new_scores, dtype=torch.float),
                img_feat,
                torch.tensor([example.q_id], dtype=torch.long))

    def __getitem__(self, index):
        if self.args.load_fast:
            example = self.features[index]
        else:
            entry = self.examples[index]
            example = self.tensorize_example(entry,
                cls_token_at_end=bool(self.args.model_type in ['xlnet']), # xlnet has a cls token at the end
                cls_token=self.tokenizer.cls_token,
                sep_token=self.tokenizer.sep_token,
                cls_token_segment_id=2 if self.args.model_type in ['xlnet'] else 0,
                pad_on_left=bool(self.args.model_type in ['xlnet']), # pad on the left for xlnet
                pad_token_segment_id=4 if self.args.model_type in ['xlnet'] else 0)
        return example

    def __len__(self):
        return len(self.examples)

    # tsv feature loading
    def load_img_tsv_features(self):
        self.check_img_feature_file()
        self.check_img_feature_offset_map()

    def check_img_feature_file(self):
        if self.img_feature_file is None:
            img_feature_path = os.path.join(self.args.img_feat_dir, '{}_img_frcnn_feats.tsv'.format(self.name))
            t_s = time.time()
            self.img_feature_file = open(img_feature_path, 'r')
            t_e = time.time()
            logger.info("Open {} image time: {}".format(self.name, (t_e - t_s)))

    def check_img_feature_offset_map(self):
        """ load the image feature offset map """
        if self.img_feat_offset_map is None:
            img_feature_path = os.path.join(self.args.img_feat_dir, '{}_img_frcnn_feats_offset_map.json'.format(self.name))
            t_s = time.time()
            self.img_feat_offset_map = json.load(open(img_feature_path))
            t_e = time.time()
            logger.info("Load {} images: {}, time: {}".format(self.name, len(self.img_feat_offset_map), (t_e - t_s)))

    def get_img_feature(self, image_id):
        """ decode the image feature """
        self.check_img_feature_file()
        self.check_img_feature_offset_map()

        if image_id in self.img_feat_offset_map:
            img_offset = self.img_feat_offset_map[image_id]
            self.img_feature_file.seek(img_offset, 0)
            arr = [s.strip() for s in self.img_feature_file.readline().split('\t')]
            num_boxes = int(arr[1])
            feat = np.frombuffer(base64.b64decode(arr[2]), dtype=np.float32).reshape((-1, self.args.img_feature_dim))
            return feat

        return None


def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size())
    one_hots.scatter_(1, logits.view(-1, 1).cpu(), 1)
    scores = (one_hots * labels.cpu())
    return scores


def trim_batch(batch):
    """ new batch func
    :param batch:
    :return:
    """
    print('batch size', len(batch))

    batch_size = len(batch)
    batch_tensors = []
    for ele in batch[0]:
        print(ele.shape, ele.size())
        zero_tensor = torch.zeros(([batch_size] + list(ele.size())))
        batch_tensors.append(zero_tensor)

    for b_id, b in enumerate(batch):
        print(b_id, len(b))
        for ele_id, ele in enumerate(b):
            print(ele_id, ele.shape)
            batch_tensors[ele_id][b_id] = ele
    return batch_tensors


def train(args, train_dataset, eval_dataset, model, tokenizer):
    """ Train the model """
    #if args.local_rank in [-1, 0]: tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, num_workers=args.workers, sampler=train_sampler, batch_size=args.train_batch_size) #, collate_fn=trim_batch)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total) # original

    if args.scheduler == "constant":
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    model, optimizer = prepare_model_optimizer(args, model, optimizer)
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step, global_loss, global_acc = 0, 0.0, 0.0
    l1_self_loss, l1_inter_loss = 0., 0.
    model.zero_grad()
    #train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args.seed, args.n_gpu)  # Added here for reproductibility (even between python 2 and 3)

    best_score = 0
    best_model = {}

    #eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True)

    for epoch in range(int(args.num_train_epochs)):
    #for epoch in train_iterator:
        #epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        logger.info("====== Epoch: %d, global_step: %d ======" % (epoch, global_step))
        total_loss = 0
        train_score = 0
        total_norm = 0
        count_norm = 0

        if args.adjust_dp and epoch>=3:
            logger.info("change droput ratio {} to 0.3".format(args.drop_out))
            if hasattr(model, 'module'):
                model.module.dropout.p = 0.3
                model.module.bert.dropout.p = 0.3
                model.module.bert.embeddings.dropout.p = 0.3
            else:
                model.dropout.p = 0.3
                model.bert.dropout.p = 0.3
                model.bert.embeddings.dropout.p = 0.3

        if args.adjust_loss and epoch>=args.adjust_loss_epoch:
            logger.info("\t change loss type from kl to bce")
            model.loss_type = 'bce'

        t_start = time.time()
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         batch[4],
                      'img_feats':      None if args.img_feature_dim == -1 else batch[5]}
            outputs = model(**inputs)

            #loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            loss, logits = outputs[:2]

            #loss = instance_bce_with_logits(logits, batch[4])

            if args.n_gpu > 1: loss = loss.mean() # mean() to average on multi-gpu parallel training

            if args.l1_loss_self_coef > 0.0:
                l1_self_loss = calculate_l1_loss(model, 'self')
                loss += l1_self_loss * args.l1_loss_self_coef

            if args.l1_loss_inter_coef > 0.0:
                l1_inter_loss = calculate_l1_loss(model, 'inter')
                loss += l1_inter_loss * args.l1_loss_inter_coef

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                total_norm += torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                count_norm += 1

            batch_acc = compute_score_with_logits(logits, batch[4]).sum(1).mean()

            mv = 0.999
            global_loss = (mv * global_loss + (1-mv) * loss.item()) if global_step != 0 else loss.item()
            global_acc = (mv * global_acc + (1-mv) * batch_acc) if global_step != 0 else batch_acc
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                if not args.pruning_steps or global_step > args.pruning_steps[-1]: 
                    scheduler.step() # don't reduce lr before finishing pruning.
                model.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f}), " \
                        "score: {:.4f} ({:.4f}), l1 loss: self {:.4f}, inter {:.4f}".format(epoch, global_step, 
                        optimizer.param_groups[0]["lr"], loss, global_loss, batch_acc, global_acc, 
                        l1_self_loss, l1_inter_loss)
                    )

            if (args.save_steps > 0 and global_step % args.save_steps == 0) or global_step == t_total:
                checkpoint_dir = save_checkpoint(model, tokenizer, args, epoch, global_step)
                if args.evaluate_during_training: 
                    eval_score = evaluate(args, model, eval_dataset, prefix=global_step)
                    if eval_score > best_score:
                        best_score = eval_score
                        best_model['epoch'] = epoch
                        best_model['global_step'] = global_step
                        best_model['model'] = copy.deepcopy(model)
                    
                    epoch_log = {'epoch': epoch, 'global_step': global_step, 
                            'eval_score': eval_score, 'best_score': best_score}
                    log_json.append(epoch_log)
                    with open(args.output_dir + '/eval_logs.json', 'w') as fp:
                        json.dump(log_json, fp)

            if global_step in args.pruning_steps:
                logger.info("Prune the model after training {} steps".format(global_step))
                if global_step == args.pruning_steps[0]:
                    # the first time to prune, count the original flops and params
                    original_flops, original_num_params = count_flops(model)
                if global_step == args.pruning_steps[-1]:
                    # the last time to prune, set the l1 loss coef to 0
                    args.l1_loss_self_coef, args.l1_loss_inter_coef = 0, 0
                    l1_self_loss, l1_inter_loss = 0, 0
                    logger.info("The last time to prune and set l1 loss coef to 0.")

                if args.evaluate_during_training:
                    # evaluate the model before pruning
                    logger.info("Evaluate the model before pruning:")
                    eval_score = evaluate(args, model, eval_dataset, prefix="original model")
                
                model = prune(args, model, logger)
                # reset optimizer
                logger.info("Resetting optimizer after pruning.")
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                    ]
                optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

                if args.scheduler == "constant":
                    scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
                elif args.scheduler == "linear":
                    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total-global_step)
                
                model, optimizer = prepare_model_optimizer(args, model, optimizer)

                if args.evaluate_during_training:
                    # evaluate the model after pruning
                    logger.info("Evaluate the model after pruning:")
                    eval_score = evaluate(args, model, eval_dataset, prefix="pruned model")
                
                pruned_flops, pruned_num_params = count_flops(model)
                logger.info(
                    "Pruning: original num of params: %.2f, after pruning %.2f (%.1f percents)",
                    original_num_params,
                    pruned_num_params,
                    pruned_num_params / original_num_params * 100,
                )
                logger.info(
                    "Pruning: original FLOPS: %.2f, after pruning %.2f (%.1f percents)",
                    original_flops,
                    pruned_flops,
                    pruned_flops / original_flops * 100,
                )

                saved_info['params'] = pruned_num_params
                saved_info['flops'] = pruned_flops
                saved_info['params_ratio'] = round(pruned_num_params / original_num_params * 100, 2)
                saved_info['flops_ratio'] = round(pruned_flops / original_flops * 100, 2)
                best_score = 0

            # if args.debug and step == 10: 
            #     args.save_steps = 10
            #     break

        t_end = time.time()
        logger.info("Progress: {}%, Time: {:.2f}".format(100*(epoch + 1) // args.num_train_epochs, t_end - t_start))

    
    logger.info("Saving the best checkpoint at epoch {}, global_step {}.".format(best_model['epoch'], best_model['global_step']))
    save_checkpoint(best_model['model'], tokenizer, args, model_name='best')

    return global_step, global_loss, best_score


def evaluate(args, model, eval_dataset=None, prefix=""):
    # assert args.local_rank in [-1, 0] # only run evaluation on the main process

    batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataloader = DataLoader(eval_dataset, num_workers=args.workers, batch_size=batch_size)

    logger.info("Running evaluation {}: Num examples = {}, Batch size = {}.".format(prefix, len(eval_dataset), batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    num_data = 0
    score = 0
    upper_bound = 0
    results_dict = {}

    for batch in eval_dataloader:
    #for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                        'labels':         batch[4],
                        'img_feats':      None if args.img_feature_dim == -1 else batch[5]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()

            # batch_score = compute_score_with_logits(logits, batch[4]).sum()
            batch_score = torch.sum(
                compute_score_with_logits(logits, batch[4]), 1)
            # update results_dict
            results_dict.update(
                {qa_ind: score for qa_ind, score in
                    zip(batch[6].view(-1).tolist(), batch_score.tolist())}
            )
            score += batch_score.sum().item()
            upper_bound += (batch[4].max(1)[0]).sum().item()
            num_data += logits.size(0)
        
        nb_eval_steps += 1

        if args.debug and nb_eval_steps == 10: break

    score = score / num_data
    upper_bound = upper_bound / num_data
    logger.warning("Eval Score: %.3f (<= %.3f)" % (100*score, 100*upper_bound))

    return score


def test(args, model, eval_dataset=None, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    label2ans = cPickle.load(open(args.label2ans_file, 'rb'))
    logger.info('label2ans: %d' % (len(label2ans)))

    results = []
    t_start = time.time()
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]: os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval
        logger.info("***** Running Test {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        for batch in eval_dataloader:
        #for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                          'labels':         None,
                          'img_feats':      None if args.img_feature_dim == -1 else batch[5]}
                outputs = model(**inputs)
                logits = outputs[0]

                val, idx = logits.max(1)
                #logger.info('idx: %s, batch[6]: %s' % (str(idx.shape), str(batch[6].shape)))

                for i in range(idx.size(0)):
                    #logger.info('idx: %d, batch: %d' % (idx[i].item(), batch[6][i].item()))
                    result = {}
                    result['question_id'] = batch[6][i].item()
                    result['answer'] = label2ans[eval_dataset.labels[idx[i].item()]] #label2ans[idx[i].item()]
                    results.append(result)

                    if len(results) % 2000 == 0:
                        logger.info("PROGRESS: {}%".format(round(100*len(results)/len(eval_dataset), 4)))
                    #logger.info('q_id: {0}, answer: {1}'.format(result['question_id'], result['answer']))

    with open(args.output_dir + ('/{}_results.json'.format(eval_dataset.name)), 'w') as fp:
        json.dump(results, fp)

    t_end = time.time()
    logger.info('# questions: %d' % (len(results)))
    logger.info('Test Time Cost: %.3f' % (t_end - t_start))


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    processor = processors[task]()
    output_mode = output_modes[task]

    label_list = processor.get_labels(args.label_file)

    t_start = time.time()
    examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)

    #img_features = torch.load(os.path.join(args.data_dir, 'val_img_feats.pt' if evaluate else 'train_img_feats.pt'))
    #img_features = torch.load(os.path.join(args.data_dir, 'val_img_frcnn_feats.pt' if evaluate else 'train_img_frcnn_feats.pt'))
    img_features = np.load(os.path.join(args.data_dir, 'val_img_frcnn_feats.npy' if evaluate else 'train_img_frcnn_feats.npy'))

    features = convert_examples_to_features_vqa(examples, img_features, label_list, args.max_img_seq_length, args.max_seq_length,
            tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)

    #if args.local_rank in [-1, 0]:
    #    logger.info("Saving features into cached file %s", cached_features_file)
    #    torch.save(features, cached_features_file)
    t_end = time.time()
    logger.info('Info: loading features using %.5f secs' % (t_end-t_start))


    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long) # batch*max_seq_len
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        labels = torch.tensor([f.label_id[0] for f in features], dtype=torch.long)
        targets = torch.tensor([target_tensor(len(label_list), f.label_id, f.score) for f in features], dtype=torch.float)

        if args.img_feature_dim > 0: # change here
            t_start = time.time()
            img_feat_np = np.zeros((labels.shape[0], args.max_img_seq_length, args.img_feature_dim))
            for f_id, f in enumerate(features):
                img_feat_np[f_id] = f.img_feat

            img_feats = torch.from_numpy(img_feat_np)

            #img_feats = torch.empty((labels.shape[0], args.max_img_seq_length, args.img_feature_dim))
            #for f_id, f in enumerate(features):
            #   img_feats[f_id] = f.img_feat

            t_end = time.time()
            logger.info('Info: convert image tensor features using %.5f secs' % (t_end - t_start))

            #img_feats = torch.stack([f.img_feat[:,-args.img_feature_dim:] for f in features])
            #img_feats = torch.stack([f.img_feat for f in features])
        #img_feats = img_feats.type(torch.long)

        #print('targets:', targets.shape)
        print('img_feats:', img_feats.shape)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    if args.img_feature_dim == -1:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, labels, targets)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, labels, targets, img_feats)
    return dataset

def target_tensor(len, labels, scores):
    """ create the target by labels and scores """
    target = [0]*len
    for id, l in enumerate(labels):
        target[l] = scores[id]

    return target

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
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

    parser.add_argument('--logging_steps', type=int, default=100, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=5000, help="Save checkpoint every X updates steps.")
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
    parser.add_argument("--debug", action='store_true', help="the debug mode to use val for train.")

    # for pruning
    parser.add_argument("--self_slimming", action='store_true', help="slimming self-attention heads in multi-head attention.")
    parser.add_argument("--inter_slimming", action='store_true', help="slimming intermediate layer in multi-head attention.")
    parser.add_argument("--l1_loss_coef", default=0., type=float, help="Coefficient for the l1 loss regularization in network")
    parser.add_argument("--l1_loss_self_coef", default=0., type=float, help="Coefficient for the l1 loss regularization in network")
    parser.add_argument("--l1_loss_inter_coef", default=0., type=float, help="Coefficient for the l1 loss regularization in network")

    parser.add_argument("--prune_before_train", action='store_true', help="Deprecated.")
    parser.add_argument("--pruning_steps", default='', type=str, help="a list of training steps when to prune, separated by comma, eg: 1e3,2e3,3e3.")
    parser.add_argument("--inter_pruning_method", default="layerwise", type=str, help="the method used to prune intermediate layers.")
    parser.add_argument("--self_pruning_method", default="layerwise", type=str, help="the method used to prune self attention heads.")
    parser.add_argument("--inter_pruning_ratio", default=0., type=float, help="pruning ratio for intermediate layers.")
    parser.add_argument("--self_pruning_ratio", default=0., type=float, help="pruning ratio for self attentions.")
    parser.add_argument("--pruning_ratio", default=0., type=float, help="pruning ratio for both self attentions and intermediate layers.")
    parser.add_argument("--pruning_strategy", default="small", type=str, help="The pruning strategy based on the coefficients: [large, random, small])")

    args = parser.parse_args()
    global saved_info
    saved_info = {'config': vars(args).copy()}

    output_dir = args.output_dir
    mkdir(output_dir)
    global logger
    logger = setup_logger("vqa", output_dir, args.local_rank)

    args.l1_loss_self_coef = args.l1_loss_self_coef or args.l1_loss_coef
    args.l1_loss_inter_coef = args.l1_loss_inter_coef or args.l1_loss_coef
    args.pruning_steps = list(map(float, args.pruning_steps.split(','))) if args.pruning_steps else []

    if args.philly:  # use philly
        logger.info('Info: Use Philly, all the output folders are reset.')
        args.output_dir = os.path.join(os.getenv('PT_OUTPUT_DIR'), args.output_dir)
        logger.info('OUTPUT_DIR:', args.output_dir)

    args.distributed = False
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
        args.distributed = True
    args.device = device

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
    config.self_slimming = args.self_slimming
    config.inter_slimming = args.inter_slimming
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

    args.config = config
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

    # Training
    if args.do_train:
        #train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        train_dataset = VQADataset(args, 'train', tokenizer) if not args.debug else eval_dataset
        tic = time.time()
        global_step, tr_loss, best_score = train(args, train_dataset, eval_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s, best score = %s", global_step, tr_loss, best_score)
        saved_info['train_time'] = round((time.time() - tic) / 3600.0, 2)
        saved_info['best_score'] = round(best_score * 100, 2)
        if args.local_rank in [-1, 0]:
            json.dump(saved_info, open(op.join(args.output_dir, 'saved_info.json'), 'w'))

    # Training on train+val
    if args.do_train_val:
        train_dataset = VQADataset(args, 'train+val', tokenizer)
        global_step, tr_loss, best_score = train(args, train_dataset, eval_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s, best score = %s", global_step, tr_loss, best_score)

    # Evaluation
    #results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            score = evaluate(args, model, eval_dataset, prefix=global_step)
            #result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            #results.update(result)

    # Test-Dev
    if args.do_test_dev and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            test(args, model, test_dev_dataset, prefix=global_step)

    # Test
    if args.do_test and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            test(args, model, test_dataset, prefix=global_step)


if __name__ == "__main__":
    main()
