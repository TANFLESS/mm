# semimm

这个目录用于放置「在 MMDetection 3.x 中用官方 DINO 套 `SemiBaseDetector`」的最小可运行工程。

## 目录说明

- `configs/dino_semibase_mini.py`
  - 训练配置主文件。
  - 已经把官方 DINO 作为 `detector` 塞到 `SemiBaseDetector` 里。
  - 包含半监督数据流（`sup / unsup_teacher / unsup_student`）和训练参数。
  - 文件开头有“用户可修改区”，用于改数据集路径、标注文件名、训练迭代数等。

- `tools/prepare_mini_coco.py`
  - 从标准 COCO 训练集中抽取少量图片（例如 100 张），构造一个可快速冒烟测试的 mini COCO。
  - 会自动生成：
    - 有标注子集（labeled）
    - 无标注子集（unlabeled，仅保留 image + categories）
    - 验证集标注（默认复用抽样全集）

- `tools/train_semibase_dino.py`
  - 训练启动脚本（MMEngine 风格）。
  - 默认读取 `configs/dino_semibase_mini.py`，可通过命令行覆盖。

- `data/`
  - 建议放 mini COCO 输出目录。

---

## 快速开始

> 假设你的原始 COCO 在：`/path/to/coco`
> 结构大致为：
> - `/path/to/coco/train2017`
> - `/path/to/coco/annotations/instances_train2017.json`

### 1) 生成 100 张图的 mini COCO

```bash
python semimm/tools/prepare_mini_coco.py \
  --src-coco-root /path/to/coco \
  --dst-root semimm/data/mini_coco \
  --num-images 100 \
  --labeled-ratio 0.5
```

### 2) 打开配置文件改路径（如果你输出路径不同）

编辑：`semimm/configs/dino_semibase_mini.py`

重点改这几项：
- `DATA_ROOT`
- `LABELED_ANN`
- `UNLABELED_ANN`
- `VAL_ANN`
- `TRAIN_IMG_PREFIX`
- `VAL_IMG_PREFIX`

### 3) 启动训练

```bash
python semimm/tools/train_semibase_dino.py \
  --config semimm/configs/dino_semibase_mini.py \
  --work-dir semimm/work_dirs/dino_semibase_mini
```

---

## 说明

1. 这个工程目标是“先跑通 DINO + SemiBaseDetector 的壳子”，不是复刻 Semi-DETR 全部创新。
2. 如果你只想快速验证流程，建议把 `max_iters` 设小（如 50~300）。
3. 若要做正式实验，再把数据换回完整 COCO，并拉长训练迭代。
