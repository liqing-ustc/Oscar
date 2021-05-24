from apex.amp.rnn_compat import has_old_rnns
from thop import profile
from copy import deepcopy
import numpy as np
import torch
from transformers.pytorch_transformers.modeling_bert import BertLayer, BertAttention

def pruning_options(parser):
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

def prune(args, model, logger=None, prune_types=['inter', 'self']):
    if hasattr(model, 'module'): model = model.module
    
    for tp in prune_types:
        if tp == 'inter':
            pruning_ratio = args.inter_pruning_ratio or args.pruning_ratio
            pruning_ratio = int(args.config.intermediate_size * pruning_ratio)
            pruning_method = args.inter_pruning_method
            layer_type = BertLayer
        elif tp == 'self':
            pruning_ratio = args.self_pruning_ratio or args.pruning_ratio
            pruning_ratio = int(args.config.num_attention_heads * pruning_ratio)
            pruning_method = args.self_pruning_method
            layer_type = BertAttention

        layers = [m for m in model.modules() if isinstance(m, layer_type)]
        slimming_coefs = [m.slimming_coef.detach().cpu().numpy().reshape(-1) for m in layers]

        if args.pruning_strategy == 'random':
            slimming_coefs = [np.random.rand(*coef.shape) for coef in slimming_coefs]
        elif args.pruning_strategy == 'large':
            slimming_coefs = [-coef for coef in slimming_coefs]
        
        if pruning_method == 'global':
            # threshold = np.quantile(np.concatenate(slimming_coefs), pruning_ratio)
            threshold = np.sort(np.concatenate(slimming_coefs))[pruning_ratio]
            threshold = [threshold] * len(slimming_coefs)
        elif pruning_method == 'layerwise':
            # threshold = [np.quantile(coef, pruning_ratio) for coef in slimming_coefs]
            threshold = [np.sort(coef)[min(pruning_ratio, len(coef) - 2)] for coef in slimming_coefs]
        else: assert False

        for m, coef, thre in zip(layers, slimming_coefs, threshold):
            prune_indice = np.where(coef <= thre)[0]
            if logger: logger.warning('Pruning {}: {}, {}'.format(tp, len(prune_indice), prune_indice[:10]))
            m.prune(prune_indice)

    return model

def calculate_l1_loss(model, tp):
    assert tp in ['inter', 'self']
    layer = BertLayer if tp == 'inter' else BertAttention
    loss = 0.0
    for m in model.modules():
        if isinstance(m, layer):
            loss += m.slimming_coef.abs().sum()
    return loss

def count_flops(model):
    if hasattr(model, 'module'): model = model.module
    batch_size, n_img_tokens, n_txt_tokens = 1, 50, 35
    input_ids = torch.ones(batch_size, n_txt_tokens, dtype=torch.int64)
    img_feat = torch.ones(batch_size, n_img_tokens, 2054, dtype=torch.float32)
    attention_mask = torch.ones(batch_size, n_img_tokens+n_txt_tokens, dtype=torch.int64)
    masked_pos = torch.ones(batch_size, n_txt_tokens, dtype=torch.int32)
    is_training = False
    inputs = (input_ids, None, attention_mask, None, None, None, img_feat)
    model = deepcopy(model)
    model.to('cpu')
    flops, params = profile(model, inputs, verbose=False) # one mul-add is counted as 1 flop
    params = sum(p.numel() for n, p in model.named_parameters())
    flops = round(flops / 1e9, 2)
    params = round(params / 1e6, 2)
    return flops, params