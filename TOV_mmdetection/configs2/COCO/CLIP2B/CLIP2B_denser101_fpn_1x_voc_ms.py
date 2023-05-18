_base_ = [
    # '../base/faster_rcnn_r50_fpn_1x_tinycoco.py',
    # '../../_base_/datasets/TinyCOCO/TinyCOCO_detection.py',
    # '../../../configs/_base_/schedules/schedule_1x.py',
    '../../../configs/_base_/default_runtime.py'
]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)  # add Group Normalization
debug = False
# model settings

#@TODO: change the whole structure into a DenseCLIP-like framework and do some specified improvements.
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
        input_resolution=992,
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
        in_channels=[256, 512, 1024, 2048 + 20], # the 4th channel is upscaled to 2068
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4,  # 5
        #  changed to 4 ,,
        norm_cfg=norm_cfg
    ),
    roi_head=dict(
        type='P2BHead', # original edition: 'P2BNet'
        num_stages=2,
        top_k=7,
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
            num_classes=20, # voc cls_num
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
                loss_weight=1,  # 0.8
                loss_type='binary_cross_entropy'),  # weight
            loss_mil2=dict(
                type='MILLoss',
                binary_ins=False,
                loss_weight=1,  # 0.8
                loss_type='gfocal_loss'),),# weight
            # loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)),
    ),
    # model training and testing settings
    train_cfg=dict(
        base_proposal=dict(
            base_scales=[16, 32, 64, 128],  # scales are after editting.
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
            gen_num_neg=100,
        ),
        rcnn=None
    ),
    test_cfg=dict(
        rpn=None,
        rcnn=None,
    ))

# dataset settings
dataset_type = 'VOCPTDataset'
data_root = '/home/junweizhou/Datasets/VOC/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),

    dict(type='Resize', img_scale=[(1504, 360), (1504, 432), (1504, 512), (1504, 648), (1504, 750), (1504, 900)]
         , multiscale_mode='value', keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_true_bboxes']),
]
test_scale = 900
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1504, test_scale) if test_scale else (1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect',
                 keys=['img', 'gt_bboxes', 'gt_labels', 'gt_anns_id', 'gt_true_bboxes']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            istrain=True,
            ann_file=[
                data_root + 'VOC2007/ImageSets/Main/train.txt'
                ,data_root + 'VOC2012/ImageSets/Main/trainval.txt'
            ],
            img_prefix=[data_root + 'VOC2007/',
                        data_root + 'VOC2012/'],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        istrain=False,
        ann_file=data_root + 'VOC2007/ImageSets/Main/train.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        istrain=False,
        ann_file=data_root + 'VOC2007/ImageSets/Main/val.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline))

check = dict(stop_while_nan=False)

# optimizer
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001,
                 paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
                                                 'text_encoder': dict(lr_mult=0.0),
                                                 'norm': dict(decay_mult=0., lr_mult = 0.0)})
                 )
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
work_dir = '../../TOV_mmdetection_cache/work_dir/voc/CLIP_voc/lr1e-4/'

evaluation = dict(
    interval=1, metric='mAP',
    save_result_file=work_dir + '_' + 'pseudo_ann',
    do_first_eval=False,  # test
    do_final_eval=True,
)


