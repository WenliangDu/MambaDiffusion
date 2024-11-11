dataset_type = 'WhuDataset'
data_root = '/data0/whu-opt-sar'
img_norm_cfg = dict(
    mean=[104.74, 96.77, 79.21, 100.92, 54.12],
    std=[171.04, 217.8, 305.58, 697.17, 2169.97],
    to_rgb=False)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadTiffImageFromFile'),
    dict(type='LoadTiffAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(256, 256), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Normalize',
        mean=[104.74, 96.77, 79.21, 100.92, 54.12],
        std=[171.04, 217.8, 305.58, 697.17, 2169.97],
        to_rgb=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadTiffImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[104.74, 96.77, 79.21, 100.92, 54.12],
                std=[171.04, 217.8, 305.58, 697.17, 2169.97],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='WhuDataset',
        data_root='/data0/whu-opt-sar',
        img_dir='cat/train',
        ann_dir='label/train',
        pipeline=[
            dict(type='LoadTiffImageFromFile'),
            dict(type='LoadTiffAnnotations', reduce_zero_label=True),
            dict(type='Resize', img_scale=(256, 256), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Normalize',
                mean=[104.74, 96.77, 79.21, 100.92, 54.12],
                std=[171.04, 217.8, 305.58, 697.17, 2169.97],
                to_rgb=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='WhuDataset',
        data_root='/data0/whu-opt-sar',
        img_dir='cat/val',
        ann_dir='label/val',
        pipeline=[
            dict(type='LoadTiffImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[104.74, 96.77, 79.21, 100.92, 54.12],
                        std=[171.04, 217.8, 305.58, 697.17, 2169.97],
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='WhuDataset',
        data_root='/data0/whu-opt-sar',
        img_dir='cat/test',
        ann_dir='label/test',
        pipeline=[
            dict(type='LoadTiffImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[104.74, 96.77, 79.21, 100.92, 54.12],
                        std=[171.04, 217.8, 305.58, 697.17, 2169.97],
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=1.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=16000, metric='mIoU', pre_eval=True)
norm_cfg = dict(type='BN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='DDP',
    timesteps=3,
    bit_scale=0.01,
    pretrained=None,
    backbone=dict(
        type='Fusion_VSSM',
        out_indices=(0, 1, 2, 3),
        in_chans=5,
        num_classes=7,
        dims=128,
        depths=(2, 2, 15, 2),
        fuse_depths=(2, 2, 2, 2),
        fuse_depthsv2=(2, 2, 4, 2),
        ssm_d_state=1,
        ssm_dt_rank='auto',
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type='v05_noz',
        fusion_typev1='fusev1_v05_noz',
        fusion_typev2='fusev2_v05',
        mlp_ratio=4.0,
        downsample_version='v3',
        patchembed_version='v2',
        drop_path_rate=0.6),
    neck=[
        dict(
            type='FPN',
            in_channels=[128, 256, 512, 1024],
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=4),
        dict(
            type='MultiStageMerging',
            in_channels=[256, 256, 256, 256],
            out_channels=256,
            kernel_size=1,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=None)
    ],
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=0,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    decode_head=dict(
        type='DeformableHeadWithTime',
        in_channels=[256],
        channels=256,
        in_index=[0],
        dropout_ratio=0.0,
        num_classes=7,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        num_feature_levels=1,
        encoder=dict(
            type='DetrTransformerEncoder',
            num_layers=6,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                use_time_mlp=True,
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention',
                    embed_dims=256,
                    num_levels=1,
                    num_heads=8,
                    dropout=0.0),
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=256,
                    feedforward_channels=1024,
                    ffn_drop=0.0,
                    act_cfg=dict(type='GELU')),
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
find_unused_parameters = True
work_dir = './work_dirs/ddp_fuse-mamba_4x4_256x256_160k_whu-fianl'
gpu_ids = range(0, 4)
auto_resume = False
