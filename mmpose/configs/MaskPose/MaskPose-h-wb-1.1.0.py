COCO_ROOT = "/path/to/COCO/"
MPII_ROOT = "/path/to/MPII/"
AIC_ROOT = "/path/to/AIC/"
OCHUMAN_ROOT = "/path/to/OCHuman/"


BATCH_SIZE = 128
COCO_NAME = "COCO"
MPII_NAME = "MPII"
AIC_NAME = "AIC"
OCHUMAN_NAME = "OCHuman"

_base_ = ['../_base_/default_runtime.py']

# resume = True
load_from = "models/pretrained/vitpose-h.pth"

# runtime
train_cfg = dict(max_epochs=210, val_interval=5)

# optimizer
custom_imports = dict(
    imports=['mmpose.engine.optim_wrappers.layer_decay_optim_wrapper'],
    allow_failed_imports=False)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=5e-4*BATCH_SIZE/64, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=32,
        layer_decay_rate=0.85,
        custom_keys={
            'bias': dict(decay_multi=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        },
    ),
    constructor='LayerDecayOptimWrapperConstructor',
    clip_grad=dict(max_norm=1., norm_type=2),
)

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='{}/AP'.format(COCO_NAME), rule='greater', max_keep_ckpts=1))

# codec settings
codec = dict(
    type='UDPHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch='huge',
        img_size=(256, 192),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.55,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        init_cfg=None,
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='models/pretrained/mae_pretrain_vit_base_20230913.pth'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=1280,
        out_channels=41,
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=False,
    ))

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(
        type='MaskBackground',
        prob=1.0,
        continue_on_failure=False,
        alpha=0.25,
        dilate_prob=0.5,
        dilate_amount=0.1,    
        erode_prob=0.5,
        erode_amount=0.5,
        patches_computation_method='blob', # random_grid or blob
    ),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='MaskBackground', continue_on_failure=False, alpha=0.25),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='PackPoseInputs')
]



coco_train_dataset = dict(
    type="CocoWholeBodyDataset",
    data_root=COCO_ROOT,
    data_mode="topdown",
    ann_file='annotations/coco_wholebody_train_v1.0.json',
    data_prefix=dict(img='train2017/'),
    pipeline=[],
    test_mode=False,
)
coco_val_dataset = dict(
    type="CocoDataset",
    data_root=COCO_ROOT,
    data_mode="topdown",
    ann_file="annotations/person_keypoints_val2017.json",
    data_prefix=dict(img='val2017/'),
    pipeline=[],
    test_mode=True,
)
mpii_train_dataset = dict(
    type="MpiiDataset",
    data_root=MPII_ROOT,
    data_mode="topdown",
    ann_file="annotations/mpii_sam_train.json",
    data_prefix=dict(img='images/'),
    pipeline=[],
    test_mode=False,
)
mpii_val_dataset = dict(
    type="MpiiDataset",
    data_root=MPII_ROOT,
    data_mode="topdown",
    ann_file="annotations/mpii_sam_val.json",
    data_prefix=dict(img='images/'),
    pipeline=[],
    test_mode=True,
)
aic_train_dataset = dict(
    type="AicDataset",
    data_root=AIC_ROOT,
    data_mode="topdown",
    ann_file="annotations/aic_sam_train.json",
    data_prefix=dict(img='images/'),
    pipeline=[],
    test_mode=False,
)
aic_val_dataset = dict(
    type="AicDataset",
    data_root=AIC_ROOT,
    data_mode="topdown",
    ann_file="annotations/aic_sam_val.json",
    data_prefix=dict(img='images/'),
    pipeline=[],
    test_mode=True,
)
ochuman_val_dataset = dict(
    type="OCHumanDataset",
    data_root=OCHUMAN_ROOT,
    data_mode="topdown",
    ann_file="../ochuman_coco_format_val_range_0.00_1.00.json",
    data_prefix=dict(img='val2017/'),
    pipeline=[],
    test_mode=True,
)

ochuman_dt_val_dataset = dict(
    type="OCHumanDetDataset",
    data_root=OCHUMAN_ROOT,
    data_mode="topdown",
    ann_file="../ochuman_coco_format_val_range_0.00_1.00.json",
    data_prefix=dict(img='val2017/'),
    bbox_file=OCHUMAN_ROOT + "../detections/rtmdet-l-ins_val.json",
    filter_cfg=dict(bbox_score_thr=0.3),
    pipeline=[],
    test_mode=True,
)

combined_val_dataset = dict(
    type='CombinedDataset',
    metainfo=dict(from_file='configs/_base_/datasets/merged_COCO_AIC_MPII_wLimbs.py'),
    datasets=[coco_val_dataset, ochuman_val_dataset, ochuman_dt_val_dataset],
    pipeline=val_pipeline,
    test_mode=True,
    keypoints_mapping=[
        {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
        9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16}, # Identity mapping for COCO as merged is based on COCO
        {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
        9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16}, # Identity mapping for OCHuman as merged is based on COCO
        {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
        9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16}, # Identity mapping for OCHuman as merged is based on COCO
    ],
)

combined_train_dataset = dict(
    type='CombinedDataset',
    metainfo=dict(from_file='configs/_base_/datasets/merged_COCO_AIC_MPII_wLimbs.py'),
    datasets=[coco_train_dataset, mpii_train_dataset, aic_train_dataset],
    pipeline=train_pipeline,
    test_mode=False,
    keypoints_mapping=[
        {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
        9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, # Identity mapping for COCO as merged is based on COCO
        17: 23, 18: 24, 19: 25, 20: 26, 21: 27, 22: 28,               # Feet kpts of COCO-wb
        95: 29, 99: 30, 103: 31, 107: 32, 111: 33,                    # Left hand kpts of COCO-wb
        116: 34, 120: 35, 124: 36, 128: 37, 132: 38,                  # Right hand kpts of COCO-wb
        71: 39, 72: 40},                                              # Mouth corners of COCO-wb   
        
        {0: 16, 1: 14, 2: 12, 3: 11, 4: 13, 5: 15, 6: 18, 7: 17, 8: 19,
        9: 20, 10: 10, 11: 8, 12: 6, 13: 5, 14: 7, 15: 9}, # MPII -> COCO and additional points
        
        {0: 6, 1: 8, 2: 10, 3: 5, 4: 7, 5: 9, 6: 12, 7: 14, 8: 16,
        9: 11, 10: 13, 11: 15, 12: 22, 13: 21}, # AIC -> COCO and additional points
    ],
)

# data loaders
train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=8,
    persistent_workers=False,
    sampler=dict(
        type='MultiSourceSampler',
        batch_size=BATCH_SIZE,
        source_ratio=[149813, 22246, 378352],
        shuffle=True,
    ),
    dataset=combined_train_dataset,
)
val_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=8,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=combined_val_dataset,
)
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='MultiDatasetEvaluator',
    metrics=[
        dict(type='CocoMetric',
            ann_file=COCO_ROOT + 'annotations/person_keypoints_val2017.json',
            prefix=COCO_NAME,
            extended=[False],
            match_by_bbox=[False],
            ignore_stats=['AP .5', 'AP .75', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)'],
        ),
        # dict(type='PCKAccuracy',
        #     prefix=MPII_NAME,
        # ),
        # dict(type='PCKAccuracy',
        #     prefix=AIC_NAME,
        # ),
        dict(type='CocoMetric',
            ann_file=OCHUMAN_ROOT + "../ochuman_coco_format_val_range_0.00_1.00.json",
            prefix=OCHUMAN_NAME,
            extended=[False],
            match_by_bbox=[False],
            ignore_stats=['AP .5', 'AP .75', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)'],
        ),
        dict(type='CocoMetric',
            ann_file=OCHUMAN_ROOT + "../ochuman_coco_format_val_range_0.00_1.00.json",
            prefix=OCHUMAN_NAME+"_det",
            extended=[False],
            match_by_bbox=[False],
            ignore_stats=['AP .5', 'AP .75', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)'],
        ),
    ],
    datasets=combined_val_dataset['datasets'],
    )
test_evaluator = val_evaluator
