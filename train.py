"""通用训练启动脚本（尽量少用命令行参数）。

使用方式：
1) 直接修改本文件顶部的【用户可改参数区】；
2) 运行：python train.py

说明：
- 该脚本优先服务你当前仓库结构：thirdparty/mmdetection-3.3.0 + configs/*
- 若后续你新增配置（如 DINO/Semi-DETR），只改 CONFIG_PATH 即可复用。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# ============================================================================
# 用户可改参数区（像 YOLOv5 一样在脚本里改）
# ============================================================================
# 训练配置文件
# 可选：'configs/SoftTeacher.py'(完整版) / 'configs/SoftTeacher_compact.py'(简略版)
CONFIG_PATH = 'configs/SoftTeacher.py'

# 工作目录：日志、权重、可视化结果会写到这里
WORK_DIR = 'work_dirs/softteacher_debug'

# 随机种子（None 表示不固定）
SEED = 42

# 是否开启 cudnn benchmark（输入分辨率相对稳定时可加速）
CUDNN_BENCHMARK = False

# 是否从 checkpoint 恢复（True 时要求 CHECKPOINT_PATH 存在）
RESUME = False
CHECKPOINT_PATH = ''

# 自动混合精度（AMP）
USE_AMP = False

# 可选：覆盖配置中的基础训练参数（None 表示不覆盖）
# 例如：MAX_ITERS = 5000 可快速 smoke test
MAX_ITERS = None
VAL_INTERVAL = None
BATCH_SIZE = None
NUM_WORKERS = None


# ============================================================================
# 内部实现
# ============================================================================
def _repo_root() -> Path:
    """返回仓库根目录（当前脚本在 train.py）。"""
    return Path(__file__).resolve().parents[1]


def _setup_python_path(repo_root: Path) -> None:
    """把 mmdetection 代码目录加入 PYTHONPATH，避免必须 pip install -e。"""
    mmdet_root = repo_root / 'thirdparty' / 'mmdetection-3.3.0'
    if not mmdet_root.exists():
        raise FileNotFoundError(
            f'未找到 mmdetection 路径: {mmdet_root}. '\
            '请确认 thirdparty/mmdetection-3.3.0 已存在。')
    if str(mmdet_root) not in sys.path:
        sys.path.insert(0, str(mmdet_root))


def _build_cfg(repo_root: Path):
    """构建并返回训练配置对象。"""
    from mmengine.config import Config

    cfg_path = repo_root / CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f'未找到配置文件: {cfg_path}')

    cfg = Config.fromfile(str(cfg_path))

    # 统一工作目录
    cfg.work_dir = str(repo_root / WORK_DIR)

    # 性能参数
    cfg.setdefault('env_cfg', {})
    cfg.env_cfg['cudnn_benchmark'] = CUDNN_BENCHMARK

    # 随机种子
    if SEED is not None:
        cfg['randomness'] = dict(seed=SEED, deterministic=False)

    # 可选覆盖：迭代数 / 验证间隔
    if MAX_ITERS is not None:
        cfg.train_cfg.max_iters = MAX_ITERS
    if VAL_INTERVAL is not None:
        cfg.train_cfg.val_interval = VAL_INTERVAL

    # 可选覆盖：batch_size / num_workers
    if BATCH_SIZE is not None:
        cfg.train_dataloader.batch_size = BATCH_SIZE
    if NUM_WORKERS is not None:
        cfg.train_dataloader.num_workers = NUM_WORKERS

    # AMP
    if USE_AMP:
        # 兼容 mmdet 3.x 常见 OptimWrapper 配置
        optim_wrapper = cfg.get('optim_wrapper', None)
        if optim_wrapper is None:
            raise ValueError('配置中缺少 optim_wrapper，无法开启 AMP。')
        optim_wrapper.type = 'AmpOptimWrapper'
        # loss_scale='dynamic' 是通用稳妥选项
        optim_wrapper.loss_scale = 'dynamic'

    # Resume 设置
    if RESUME:
        cfg.resume = True
        if CHECKPOINT_PATH:
            ckpt = repo_root / CHECKPOINT_PATH
            if not ckpt.exists():
                raise FileNotFoundError(f'找不到恢复权重: {ckpt}')
            cfg.load_from = str(ckpt)

    return cfg


def main() -> None:
    repo_root = _repo_root()
    os.chdir(repo_root)
    _setup_python_path(repo_root)

    # 延迟 import，确保 sys.path 已注入 mmdet 根目录
    from mmengine.runner import Runner

    cfg = _build_cfg(repo_root)

    print('=' * 80)
    print('Training launcher')
    print(f'CONFIG_PATH: {CONFIG_PATH}')
    print(f'WORK_DIR   : {cfg.work_dir}')
    print(f'RESUME     : {RESUME}')
    print(f'USE_AMP    : {USE_AMP}')
    print('=' * 80)

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
