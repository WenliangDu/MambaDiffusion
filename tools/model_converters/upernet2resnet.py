# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmcv
import torch
from mmcv.runner import CheckpointLoader


def convert_upernet(ckpt):
    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('backbone'):
            new_key = k.replace('backbone.', '')
            new_ckpt[new_key] = v
        elif k.startswith('patch_embed'):
            new_key = k.replace('patch_embed.proj', 'patch_embed.projection')
            new_ckpt[new_key] = v
        else:
            new_key = k
            new_ckpt[new_key] = v
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in official pretrained beit models to'
        'MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    weight = convert_upernet(state_dict)
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    result = dict(meta={}, state_dict=weight)
    torch.save(result, args.dst)


if __name__ == '__main__':
    main()
