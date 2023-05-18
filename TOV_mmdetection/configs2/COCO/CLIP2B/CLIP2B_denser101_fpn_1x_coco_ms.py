_base_ = [
    # '../base/faster_rcnn_r50_fpn_1x_tinycoco.py',
    # '../../_base_/datasets/TinyCOCO/TinyCOCO_detection.py',
    # '../../../configs/_base_/schedules/schedule_1x.py',
    '../../../configs/_base_/default_runtime.py'
]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)  # add Group Normalization
debug = False
# model settings

num_stages = 2
model = dict(
    type='CLIP2B',
    pretrained='pretrained/RN101.pt',
    context_length=5,
    clip_head=False,
    seg_loss=False,
    point_ambience = False,
    gen_range = None,
    text_dim=512,
    backbone=dict(
        type='CLIPResNetWithAttention',
        layers=[3, 4, 23, 3],
        output_dim=512,
        input_resolution=1344,
        style='pytorch'),
    text_encoder=dict(
        type='CLIPTextContextEncoder',
        context_length=13,
        embed_dim=512,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        style='pytorch'),
    context_decoder=dict(
        type='ContextDecoder',
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        visual_dim=512,
        dropout=0.1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048 + 80],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4,  # 5
        #  changed to 4 ,,
        norm_cfg=norm_cfg
    ),
    roi_head=dict(
        type='P2BHead',
        num_stages=2,
        top_k=4,
        with_atten=False,
        # stage_loss_weights=[1] * num_stages,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCInstanceMILHead',
            num_stages=2,
            with_loss_pseudo=False,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            num_ref_fcs=0,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_type='MIL',
            loss_mil1=dict(
                type='MILLoss',
                binary_ins=False,
                loss_weight=0.25,
                loss_type='binary_cross_entropy'),
            loss_mil2=dict(
                type='MILLoss',
                binary_ins=False,
                loss_weight=0.25,
                loss_type='gfocal_loss'),),
            # loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)),
    ),
    # model training and testing settings
    train_cfg=dict(
        base_proposal=dict(
            base_scales=[4, 8, 16, 32, 64, 128],
            base_ratios=[1 / 3, 1 / 2, 1 / 1.5, 1.0, 1.5, 2.0, 3.0],
            shake_ratio=None,
            cut_mode='symmetry',  # 'clamp',
            gen_num_neg=0),
        fine_proposal=dict(
            gen_proposal_mode='fix_gen',
            cut_mode=None,
            shake_ratio=[0.1],
            base_ratios=[1, 1.2, 1.3, 0.8, 0.7],
            # gen_num_per_box=10,
            iou_thr=0.3,
            gen_num_neg=500,
        ),
        rcnn=None
    ),
    test_cfg=dict(
        rpn=None,
        rcnn=None,
    ))

# dataset settings
dataset_type = 'CocoFmtDataset'
data_root = '/home/junweizhou/Datasets/COCO/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize',
        img_scale=[(2000, 480), (2000, 576), (2000, 688), (2000, 864), (2000, 1000), (2000, 1200)],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5) if not debug else dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_true_bboxes']),
]

# test_scale = 1200
test_scale = 1200
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2000, test_scale) if test_scale else (1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect',
                 keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_anns_id', 'gt_true_bboxes']),
        ])
]
data = dict(
    samples_per_gpu=1,  # 2
    workers_per_gpu=2,
    shuffle=False if debug else None,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "annotations_qc_pt/instances_train2017_coarse.json",
        img_prefix=data_root + 'train2017/',  # 'train2017/',

        pipeline=train_pipeline,
    ),
    val=dict(
        samples_per_gpu=2,
        type=dataset_type,
        ann_file=data_root + "annotations_qc_pt/instances_train2017_coarse.json",
        img_prefix=data_root + 'train2017/',  # 'train2017/',
        pipeline=test_pipeline,
        test_mode=False,  # modified
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))

check = dict(stop_while_nan=False)

# optimizer
optimizer = dict(type='AdamW', lr=0.00005, weight_decay=0.0002,  # changing for 2e-5 changing weight decay for a try
                 paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
                                                 'text_encoder': dict(lr_mult=0.0),
                                                 'norm': dict(decay_mult=0.)}) # learning rate to be modified.
                 # waiting for further validation.
                 ) # when using 8 gpus and 2 samples per gpu  lr being 1e-4
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
work_dir = '../'

evaluation = dict(
    interval=1, metric='bbox',
    # save_result_file=work_dir + '_' + str(test_scale) + '_latest_result_CLIP.json',
    do_first_eval=False,  # test
    do_final_eval=True,
)


