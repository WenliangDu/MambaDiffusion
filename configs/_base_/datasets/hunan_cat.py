# dataset settings
dataset_type = 'HunanDataset'
data_root = '/data0/Hunan'
img_norm_cfg = dict(
    mean=[1400.32, 1117.23, 994.44, 787.33, 1016.67, 1754.84, 2082.32, 1973.73, 2264.01, 502.14, 6.38, 1482.67, 818.22, -10.33, -16.48], 
    std=[9664.63, 14092.01, 20028.46, 48678.92, 44938.45, 184570.05, 316813.94, 320642.53, 436656.31, 16580.50, 11.98, 250519.375, 136984.2, 20.4, 18.79], 
    to_rgb=False)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadTiffImageFromFile'),
    dict(type='LoadTiffAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(256, 256), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadTiffImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='cat1/train',
        ann_dir='lbl/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='cat1/val',
        ann_dir='lbl/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='cat1/test',
        ann_dir='lbl/test',
        pipeline=test_pipeline),
)
