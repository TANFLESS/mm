# mm

## 仓库说明（讲解阶段）

当前仓库包含两个核心代码库：

1. `mmdetection-3.3.0/`：MMDetection 3.x 系列源码（较新架构，基于 MMEngine + MMCV 2.x 风格）。
2. `Semi-DETR-main/`：基于 MMDetection 2.16.0 的 Semi-DETR 实现（CVPR 2023），内部包含 `thirdparty/mmdetection` 2.x 版本依赖和自定义 `detr_od` / `detr_ssod` 模块。

本次仅做结构讲解，不修改算法代码。
