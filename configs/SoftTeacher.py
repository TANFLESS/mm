"""SoftTeacher 实验配置（尽量保持与你原文件结构一致）。

说明：
1) 该文件保留了你原本“在本文件里直接写 pipeline + dataset + model”的组织方式；
2) 同时补齐了你原文件里缺失的关键变量定义，避免运行时报 NameError；
3) 将“你最常改的数据集路径相关参数”集中放到文件顶部，方便直接改。
"""

# ============================================================================
# 0) 你最常改的路径参数（放在最顶端，方便直接修改）
# ============================================================================
# COCO 数据根目录（建议保持以 / 结尾）
DATA_ROOT = 'data/coco/'

# 有监督数据（带框）标注文件与图片目录
LABELED_ANN_FILE = 'annotations/instances_train2017.json'
LABELED_IMG_PREFIX = 'train2017/'

# 无监督数据（无框/空标注）标注文件与图片目录
UNLABELED_ANN_FILE = 'annotations/instances_unlabeled2017.json'
UNLABELED_IMG_PREFIX = 'unlabeled2017/'

# 验证集（可按需修改）
VAL_ANN_FILE = 'annotations/instances_val2017.json'
VAL_IMG_PREFIX = 'val2017/'


# ============================================================================
# 1) 继承基础配置
# ============================================================================
# 这里仍然继承官方 model/runtime/dataset base：
# - model: faster-rcnn_r50_fpn
# - runtime: 默认训练 hooks/logger/runtime
# - dataset: semi_coco_detection（我们会在当前文件里按你的风格覆盖）
_base_ = [
    '../thirdparty/mmdetection-3.3.0/configs/_base_/models/faster-rcnn_r50_fpn.py',
    '../thirdparty/mmdetection-3.3.0/configs/_base_/default_runtime.py',
    '../thirdparty/mmdetection-3.3.0/configs/_base_/datasets/semi_coco_detection.py'
]


# ============================================================================
# 2) 数据与增强基础参数（你原文件缺失的关键变量）
# ============================================================================
dataset_type = 'CocoDataset'
data_root = DATA_ROOT
backend_args = None

# RandAugment 的颜色变换空间
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

# RandAugment 的几何变换空间
geometric = [
    [dict(type='Rotate')],
    [dict(type='ShearX')],
    [dict(type='ShearY')],
    [dict(type='TranslateX')],
    [dict(type='TranslateY')],
]

# 多尺度训练范围
scale = [(1333, 400), (1333, 1200)]

# MultiBranch 会把 batch 显式拆分为这 3 个分支
branch_field = ['sup', 'unsup_teacher', 'unsup_student']

# dataloader 基础参数
batch_size = 5
num_workers = 5


# ============================================================================
# 3) 多分支数据流程（保持你原本写法）
# ============================================================================
# （1）有监督分支：用于 student 监督学习
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

# （2）无监督 teacher 分支：弱增强，用于 teacher 生成伪标签
weak_pipeline = [
    dict(type='RandomResize', scale=scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

# （3）无监督 student 分支：强增强，用于 student 学习伪标签
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

# 无监督样本先 LoadImage/LoadEmptyAnnotations，再拆 teacher/student 两路
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

# 验证/测试 pipeline
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


# ============================================================================
# 4) 数据集与 dataloader
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
# 5) 半监督模型（保持官方 SoftTeacher 参数）
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
        # teacher 参数通过 EMA 更新，不参与梯度训练
        freeze_teacher=True,
        # 监督与无监督损失权重
        sup_weight=1.0,
        unsup_weight=4.0,
        # 伪标签筛选阈值
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_thr=0.9,
        cls_pseudo_thr=0.9,
        # 回归分支不确定性过滤阈值
        reg_pseudo_thr=0.02,
        # 对伪框进行 jitter 来估计回归稳定性
        jitter_times=10,
        jitter_scale=0.06,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='teacher'))


# ============================================================================
# 6) 训练调度与 hooks
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

# 保存与日志策略
# - by_epoch=False: 与 IterBasedTrainLoop 对齐
# - interval=10000: 每 1w iter 存一次
# - max_keep_ckpts=2: 防止磁盘占满
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=10000, max_keep_ckpts=2))
log_processor = dict(by_epoch=False)

# MeanTeacherHook：核心的 teacher EMA 更新机制
custom_hooks = [dict(type='MeanTeacherHook')]
