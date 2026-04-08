"""SoftTeacher（完整版，显式展开版）。

目标：
- 保留“所有关键部件都在一个文件里看得见”的体验；
- 明确标注哪些是文档展示内容、哪些是为可运行补充；
- 明确标注主要来源文件，便于你后续追溯。

主要来源：
1) 文档：docs user_guides/semi_det（展示半监督核心流程）
2) 官方数据基线：thirdparty/mmdetection-3.3.0/configs/_base_/datasets/semi_coco_detection.py
3) 官方SoftTeacher配置：thirdparty/mmdetection-3.3.0/configs/soft_teacher/
   soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco.py
"""

# ============================================================================
# [补充] 0) 便于修改的路径参数（文档里通常不会集中写在最顶部）
# 来源：你的使用习惯 + 官方配置字段
# ============================================================================
DATA_ROOT = 'data/coco/'
LABELED_ANN_FILE = 'annotations/instances_train2017.json'
LABELED_IMG_PREFIX = 'train2017/'
UNLABELED_ANN_FILE = 'annotations/instances_unlabeled2017.json'
UNLABELED_IMG_PREFIX = 'unlabeled2017/'
VAL_ANN_FILE = 'annotations/instances_val2017.json'
VAL_IMG_PREFIX = 'val2017/'


# ============================================================================
# [文档/官方示例] 1) 继承基础配置
# 来源：soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco.py
# ============================================================================
_base_ = [
    '../thirdparty/mmdetection-3.3.0/configs/_base_/models/faster-rcnn_r50_fpn.py',
    '../thirdparty/mmdetection-3.3.0/configs/_base_/default_runtime.py',
    '../thirdparty/mmdetection-3.3.0/configs/_base_/datasets/semi_coco_detection.py'
]


# ============================================================================
# [文档摘录 + 可运行补齐] 2) dataset/pipeline 基础变量
# 来源：_base_/datasets/semi_coco_detection.py
# 说明：这些变量如果不显式给出，很多“文档片段”粘贴后会报 NameError。
# ============================================================================
dataset_type = 'CocoDataset'
data_root = DATA_ROOT
backend_args = None

color_space = [
    [dict(type='ColorTransform')],
    [dict(type='AutoContrast')],
    [dict(type='Equalize')],
    [dict(type='Sharpness')],
    [dict(type='Posterize')],
    [dict(type='Solarize')],
    [dict(type='Color')],
    [dict(type='Contrast')],
    [dict(type='Brightness')],
]

geometric = [
    [dict(type='Rotate')],
    [dict(type='ShearX')],
    [dict(type='ShearY')],
    [dict(type='TranslateX')],
    [dict(type='TranslateY')],
]

scale = [(1333, 400), (1333, 1200)]
branch_field = ['sup', 'unsup_teacher', 'unsup_student']

# [补充] dataloader 常用默认值（来自官方 base）
batch_size = 5
num_workers = 5


# ============================================================================
# [文档展示核心] 3) 三分支 pipeline（半监督最关键部分）
# 来源：_base_/datasets/semi_coco_detection.py + semi_det 文档示例
# ============================================================================
# 有监督分支：送入 student 做监督损失
sup_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandAugment', aug_space=color_space, aug_num=1),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        sup=dict(type='PackDetInputs'))
]

# 无监督 teacher 分支：弱增强，teacher 在这路上产伪标签
weak_pipeline = [
    dict(type='RandomResize', scale=scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

# 无监督 student 分支：强增强，student 学习伪标签
strong_pipeline = [
    dict(type='RandomResize', scale=scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomOrder',
        transforms=[
            dict(type='RandAugment', aug_space=color_space, aug_num=1),
            dict(type='RandAugment', aug_space=geometric, aug_num=1),
        ]),
    dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

# 无监督样本进入后拆成 teacher/student 两路
unsup_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadEmptyAnnotations'),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
    )
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


# ============================================================================
# [文档有示例，以下是可运行补齐] 4) dataset/dataloader/evaluator
# 来源：_base_/datasets/semi_coco_detection.py
# ============================================================================
labeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=LABELED_ANN_FILE,
    data_prefix=dict(img=LABELED_IMG_PREFIX),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=sup_pipeline,
    backend_args=backend_args)

unlabeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=UNLABELED_ANN_FILE,
    data_prefix=dict(img=UNLABELED_IMG_PREFIX),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=unsup_pipeline,
    backend_args=backend_args)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(
        type='GroupMultiSourceSampler',
        batch_size=batch_size,
        source_ratio=[1, 4]),
    dataset=dict(
        type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=VAL_ANN_FILE,
        data_prefix=dict(img=VAL_IMG_PREFIX),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + VAL_ANN_FILE,
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator


# ============================================================================
# [文档展示核心 + 官方soft-teacher config] 5) 模型
# 来源：soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco.py
# ============================================================================
detector = _base_.model
detector.data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    bgr_to_rgb=False,
    pad_size_divisor=32)
detector.backbone = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN', requires_grad=False),
    norm_eval=True,
    style='caffe',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='open-mmlab://detectron2/resnet50_caffe'))

model = dict(
    _delete_=True,
    type='SoftTeacher',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=4.0,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_thr=0.9,
        cls_pseudo_thr=0.9,
        reg_pseudo_thr=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='teacher'))


# ============================================================================
# [补充但基本等同官方] 6) 训练策略与 hooks
# 来源：soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco.py
# ============================================================================
train_cfg = dict(type='IterBasedTrainLoop', max_iters=180000, val_interval=5000)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=180000,
        by_epoch=False,
        milestones=[120000, 160000],
        gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=10000, max_keep_ckpts=2))
log_processor = dict(by_epoch=False)
custom_hooks = [dict(type='MeanTeacherHook')]
