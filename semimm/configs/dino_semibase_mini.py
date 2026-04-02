"""
DINO + SemiBaseDetector 最小可运行配置（MMDetection 3.x）
=====================================================

本配置用于快速验证：
1) 使用官方 DINO 作为 detector；
2) 外层套 SemiBaseDetector 做 teacher-student 半监督训练；
3) 在一个很小的 mini COCO 数据集上做冒烟测试。

你最常改的是下面“用户可修改区”。
"""

# -------------------------------
# 用户可修改区（重点）
# -------------------------------

# mini COCO 根目录。
# 由 semimm/tools/prepare_mini_coco.py 生成后，通常是：semimm/data/mini_coco/
DATA_ROOT = 'semimm/data/mini_coco/'

# 标注文件（相对于 DATA_ROOT）
LABELED_ANN = 'annotations/instances_train2017_mini_labeled.json'
UNLABELED_ANN = 'annotations/instances_train2017_mini_unlabeled.json'
VAL_ANN = 'annotations/instances_val2017_mini.json'

# 图片前缀（相对于 DATA_ROOT）
TRAIN_IMG_PREFIX = 'train2017/'
VAL_IMG_PREFIX = 'train2017/'

# 训练超参（为了快速测试，默认设得较小）
BATCH_SIZE = 2
NUM_WORKERS = 2
MAX_ITERS = 200
VAL_INTERVAL = 100

# 半监督关键超参
SOURCE_RATIO = [1, 1]  # 每个 batch 中 labeled:unlabeled 采样比例
CLS_PSEUDO_THR = 0.9
UNSUP_WEIGHT = 1.0


# -------------------------------
# 基础配置继承
# -------------------------------
# 这里复用：
# - 官方 DINO 配置（模型结构、损失等）
# - 官方 semi 数据管线模板（sup/unsup 三分支）
# - 默认 runtime
_base_ = [
    '../../mmdetection-3.3.0/configs/dino/dino-4scale_r50_8xb2-12e_coco.py',
    '../../mmdetection-3.3.0/configs/_base_/datasets/semi_coco_detection.py',
    '../../mmdetection-3.3.0/configs/_base_/default_runtime.py'
]


# -------------------------------
# 模型：官方 DINO 外套 SemiBaseDetector
# -------------------------------
# detector 直接拿 DINO 基础配置。
detector = _base_.model

# SemiBaseDetector 需要 MultiBranchDataPreprocessor 来处理多分支输入。
model = dict(
    _delete_=True,
    type='SemiBaseDetector',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=UNSUP_WEIGHT,
        cls_pseudo_thr=CLS_PSEUDO_THR,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='teacher')
)


# -------------------------------
# 数据集与 DataLoader
# -------------------------------
# 在官方 semi_coco_detection.py 的基础上覆写路径。
labeled_dataset = _base_.labeled_dataset
labeled_dataset.data_root = DATA_ROOT
labeled_dataset.ann_file = LABELED_ANN
labeled_dataset.data_prefix = dict(img=TRAIN_IMG_PREFIX)

unlabeled_dataset = _base_.unlabeled_dataset
unlabeled_dataset.data_root = DATA_ROOT
unlabeled_dataset.ann_file = UNLABELED_ANN
unlabeled_dataset.data_prefix = dict(img=TRAIN_IMG_PREFIX)

train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=False,
    sampler=dict(
        type='GroupMultiSourceSampler',
        batch_size=BATCH_SIZE,
        source_ratio=SOURCE_RATIO),
    dataset=dict(type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset])
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=DATA_ROOT,
        ann_file=VAL_ANN,
        data_prefix=dict(img=VAL_IMG_PREFIX),
        test_mode=True,
        pipeline=_base_.test_pipeline,
        backend_args=None))

# 测试集这里复用验证集（快速冒烟场景）
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=DATA_ROOT + VAL_ANN,
    metric='bbox',
    format_only=False,
    backend_args=None)

test_evaluator = val_evaluator


# -------------------------------
# 训练/验证循环
# -------------------------------
train_cfg = dict(type='IterBasedTrainLoop', max_iters=MAX_ITERS, val_interval=VAL_INTERVAL)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')


# -------------------------------
# 优化器与学习率（沿用 DINO 常用设置，适度简化）
# -------------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)

# 短跑测试用调度策略
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=20),
    dict(type='MultiStepLR', begin=0, end=MAX_ITERS, by_epoch=False,
         milestones=[int(MAX_ITERS * 0.7), int(MAX_ITERS * 0.9)], gamma=0.1)
]


# -------------------------------
# Hooks / 日志 / 断点
# -------------------------------
# MeanTeacherHook 负责 teacher 参数 EMA 更新。
custom_hooks = [dict(type='MeanTeacherHook', momentum=0.001, interval=1)]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=VAL_INTERVAL, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=10)
)

log_processor = dict(by_epoch=False)

# 给这个实验一个默认工作目录名（可在启动命令中覆盖）
work_dir = 'semimm/work_dirs/dino_semibase_mini'
