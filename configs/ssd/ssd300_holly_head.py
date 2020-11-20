_base_ = [
    '../_base_/models/ssd300.py', '../_base_/datasets/holly_wood_head.py',
    '../_base_/default_runtime.py'
]
model = dict(bbox_head=dict(num_classes=1))
# optimizer
optimizer = dict(type='SGD', lr=0.008, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[16, 20])
# runtime settings
total_epochs = 24
log_config = dict(interval=20)
evaluation = dict(interval=1, metric=['mAP'],iou_thr=0.5)
