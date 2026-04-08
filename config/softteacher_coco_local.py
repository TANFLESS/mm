"""
SoftTeacher 本地训练配置（面向你当前仓库结构）
=================================================

这个配置的目标是：
1) 保持 MMDetection 官方 SoftTeacher 配置逻辑；
2) 把你本地 data 目录下已经切好的 semi annotations 接进来；
3) 补齐文档里没写清楚的“引用关系”和“本地路径可改入口”；
4) 注释尽量详细，方便你后续自己改。

使用前你主要修改【用户可修改区】即可。
"""

from pathlib import Path

# =========================================================
# 用户可修改区（只改这里，其他部分尽量别动）
# =========================================================

# 你的 COCO 数据根目录（建议和你说的一样放在仓库 data 里）
# 目标结构示例：
# data/coco/
#   ├── train2017/
#   ├── val2017/
#   ├── unlabeled2017/   # 如果你用 setting-2 才需要
#   └── annotations/
DATA_ROOT = 'data/coco/'

# 你已经按 MMDet 文档切好的半监督标注文件（setting-1 示例）
# 例如：instances_train2017.1@10.json 与 instances_train2017.1@10-unlabeled.json
LABELED_ANN = 'annotations/semi_anns/instances_train2017.1@10.json'
UNLABELED_ANN = 'annotations/semi_anns/instances_train2017.1@10-unlabeled.json'

# 训练图像目录（setting-1 通常都来自 train2017）
LABELED_IMG_PREFIX = 'train2017/'
UNLABELED_IMG_PREFIX = 'train2017/'

# 验证集（标准 COCO 验证）
VAL_ANN = 'annotations/instances_val2017.json'
VAL_IMG_PREFIX = 'val2017/'

# 训练总迭代与评估间隔（可先小一点做冒烟）
MAX_ITERS = 180000
VAL_INTERVAL = 5000

# batch 与采样比例
BATCH_SIZE = 5
NUM_WORKERS = 5
SOURCE_RATIO = [1, 4]  # labeled : unlabeled


# =========================================================
# 自动定位 MMDetection 源码目录（适配两种仓库摆放方式）
# =========================================================
# 你说已经把参考库搬到 thirdparty 下了；
# 同时兼容当前仓库里仍存在 mmdetection-3.3.0 的情况。
_repo_root = Path(__file__).resolve().parents[1]

if (_repo_root / 'thirdparty' / 'mmdetection').exists():
    _mmdet_rel = '../thirdparty/mmdetection'
elif (_repo_root / 'mmdetection-3.3.0').exists():
    _mmdet_rel = '../mmdetection-3.3.0'
else:
    raise RuntimeError(
        '未找到 MMDetection 源码目录：期望 thirdparty/mmdetection 或 mmdetection-3.3.0')


# =========================================================
# 配置继承关系（这就是文档里常说但没完全展开的“引用”）
# =========================================================
# 这里直接继承官方 10% soft-teacher 配置：
# - 模型结构（SoftTeacher + Faster R-CNN）
# - semi_train_cfg / semi_test_cfg
# - MeanTeacherHook / loop / scheduler 等
_base_ = [f'{_mmdet_rel}/configs/soft_teacher/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco.py']


# =========================================================
# 覆写数据路径（将官方示例改成你本地 data）
# =========================================================
# 1) 先覆写基础 data_root
labeled_dataset = _base_.labeled_dataset
labeled_dataset.data_root = DATA_ROOT
labeled_dataset.ann_file = LABELED_ANN
labeled_dataset.data_prefix = dict(img=LABELED_IMG_PREFIX)

unlabeled_dataset = _base_.unlabeled_dataset
unlabeled_dataset.data_root = DATA_ROOT
unlabeled_dataset.ann_file = UNLABELED_ANN
unlabeled_dataset.data_prefix = dict(img=UNLABELED_IMG_PREFIX)

# 2) train_dataloader 使用两者拼接
train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    sampler=dict(
        type='GroupMultiSourceSampler',
        batch_size=BATCH_SIZE,
        source_ratio=SOURCE_RATIO),
    dataset=dict(type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset])
)

# 3) val/test 路径覆写
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=DATA_ROOT,
        ann_file=VAL_ANN,
        data_prefix=dict(img=VAL_IMG_PREFIX),
        test_mode=True,
        pipeline=_base_.test_dataloader.dataset.pipeline,
        backend_args=None))

test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', ann_file=DATA_ROOT + VAL_ANN, metric='bbox', format_only=False)
test_evaluator = val_evaluator


# =========================================================
# 训练节奏覆写（便于本地调试）
# =========================================================
train_cfg = dict(type='IterBasedTrainLoop', max_iters=MAX_ITERS, val_interval=VAL_INTERVAL)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

# checkpoint 间隔通常和评估间隔一致，方便对齐观察
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=VAL_INTERVAL, max_keep_ckpts=3),
)

# 给一个本地默认 work_dir（命令行可覆盖）
work_dir = 'work_dirs/softteacher_coco_local'
