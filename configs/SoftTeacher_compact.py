"""SoftTeacher（简略版，尽可能复用官方配置）。

思路：
- 最大化引用官方，最小化自定义代码；
- 只覆盖你最常改的字段（数据路径与前缀）；
- 适合长期维护与跟进 mmdetection 升级。

基底来源：
- thirdparty/mmdetection-3.3.0/configs/soft_teacher/
  soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco.py
"""

# 你常改的路径参数（顶置）
DATA_ROOT = 'data/coco/'
LABELED_ANN_FILE = 'annotations/instances_train2017.json'
LABELED_IMG_PREFIX = 'train2017/'
UNLABELED_ANN_FILE = 'annotations/instances_unlabeled2017.json'
UNLABELED_IMG_PREFIX = 'unlabeled2017/'
VAL_ANN_FILE = 'annotations/instances_val2017.json'
VAL_IMG_PREFIX = 'val2017/'

# 直接继承官方可运行 soft-teacher 配置
_base_ = [
    '../thirdparty/mmdetection-3.3.0/configs/soft_teacher/'
    'soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco.py'
]

# 覆盖 base dataset 的根目录（base 里默认也是 data/coco/，这里显式化）
data_root = DATA_ROOT

# 覆盖训练数据集（labeled / unlabeled）
labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset

labeled_dataset.data_root = data_root
labeled_dataset.ann_file = LABELED_ANN_FILE
labeled_dataset.data_prefix = dict(img=LABELED_IMG_PREFIX)

unlabeled_dataset.data_root = data_root
unlabeled_dataset.ann_file = UNLABELED_ANN_FILE
unlabeled_dataset.data_prefix = dict(img=UNLABELED_IMG_PREFIX)

train_dataloader = dict(dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))

# 覆盖验证集路径
val_dataset_base = _base_.val_dataloader.dataset
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=VAL_ANN_FILE,
        data_prefix=dict(img=VAL_IMG_PREFIX),
        pipeline=val_dataset_base.pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + VAL_ANN_FILE)
test_evaluator = val_evaluator
