#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SoftTeacher 训练启动脚本（面向当前仓库）
======================================

作用：
1) 自动定位 mmdetection 源码目录；
2) 调用官方 tools/train.py 启动训练；
3) 提供更友好的默认参数和中文提示。
"""

import argparse
import subprocess
import sys
from pathlib import Path


def detect_mmdet_root(repo_root: Path) -> Path:
    """自动检测 MMDetection 根目录。"""
    candidates = [
        repo_root / 'thirdparty' / 'mmdetection',
        repo_root / 'mmdetection-3.3.0',
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError('未找到 MMDetection 目录：thirdparty/mmdetection 或 mmdetection-3.3.0')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='启动 SoftTeacher 训练')
    parser.add_argument(
        '--config',
        default='config/softteacher_coco_local.py',
        help='训练配置文件路径（默认使用本项目 softteacher 配置）')
    parser.add_argument(
        '--work-dir',
        default=None,
        help='输出目录（日志、权重）。不传则用配置里的 work_dir')
    parser.add_argument(
        '--resume',
        default=None,
        help='恢复训练：传 checkpoint 路径；传 auto 表示自动恢复')
    parser.add_argument(
        '--launcher',
        default='none',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        help='分布式启动器')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        default=None,
        help='配置覆盖，如 data_root=data/coco/ train_cfg.max_iters=1000')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    mmdet_root = detect_mmdet_root(repo_root)

    train_script = mmdet_root / 'tools' / 'train.py'
    config_path = repo_root / args.config

    if not config_path.exists():
        raise FileNotFoundError(f'配置文件不存在: {config_path}')

    cmd = [
        sys.executable,
        str(train_script),
        str(config_path),
        '--launcher',
        args.launcher,
    ]

    if args.work_dir:
        cmd.extend(['--work-dir', args.work_dir])

    if args.resume is not None:
        if args.resume == 'auto':
            cmd.extend(['--resume'])
        else:
            cmd.extend(['--resume', args.resume])

    if args.cfg_options:
        cmd.extend(['--cfg-options', *args.cfg_options])

    print('即将执行命令：')
    print(' '.join(cmd))

    subprocess.run(cmd, check=True, cwd=repo_root)


if __name__ == '__main__':
    main()
