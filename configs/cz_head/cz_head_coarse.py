input_size = 256
model = dict(
    type='CZ_PersonHeadCoarseDetector',
    pretrained=None,#'/home/lhw/m2_disk/work_dir/cz_head_coarse_224/slim_Final.pth',
    backbone=dict(
        type='CZ_PersonHeadBackbone'),
    neck=None,
    bbox_head=dict(
        type='CZ_CoarseHead',
        num_classes=1,
        in_channels=[128,256,256]
       ))
cudnn_benchmark = True
# train_cfg=dict(
#     assigner=dict(
#             type='GridAssigner', pos_iou_thr=0.45, neg_iou_thr=0.1, min_pos_iou=0.1
#     ))

train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.45,
        neg_iou_thr=0.2,
        min_pos_iou=0.0,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    smoothl1_beta=1.,
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)

test_cfg = dict(
    nms_pre=200,
    min_bbox_size=0,
    score_thr=0.1,
    conf_thr=0.1,
    nms=dict(type='nms', iou_threshold=0.3),
    max_per_img=10)
dataset_type = 'HollyWoodHeadDataset'
data_root = '/home/lhw/m2_disk/data/HollyWoodHead/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=24,
        contrast_range=(0.6, 1.4),
        saturation_range=(0.6, 1.4),
        hue_delta=12),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1,0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(256, 160), keep_ratio=False),
    dict(
        type='Normalize',**img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 160),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(
                type='Normalize',**img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=3,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='HollyWoodHeadDataset',
            ann_file=
            '/home/lhw/m2_disk/data/HollyWoodHead//ImageSets/Main/train.txt',
            img_prefix='/home/lhw/m2_disk/data/HollyWoodHead/',
            min_size=17,
            pipeline=train_pipeline
        )),
    val=dict(
        type='HollyWoodHeadDataset',
        ann_file='/home/lhw/m2_disk/data/HollyWoodHead//ImageSets/Main/val.txt',
        img_prefix='/home/lhw/m2_disk/data/HollyWoodHead/',
        pipeline= test_pipeline),
    test=dict(
        type='HollyWoodHeadDataset',
        ann_file=
        '/home/lhw/m2_disk/data/HollyWoodHead//ImageSets/Main/test.txt',
        img_prefix='/home/lhw/m2_disk/data/HollyWoodHead/',
        pipeline=test_pipeline)
    )

checkpoint_config = dict(interval=1)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    step=[10, 16])
total_epochs = 20
evaluation = dict(interval=1, metric=['mAP'],iou_thr=0.5)
load_from = '/home/lhw/m2_disk/work_dir/cz_head_coarse_256/latest.pth'
