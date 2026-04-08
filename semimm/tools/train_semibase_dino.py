#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINO + SemiBaseDetector 训练启动脚本。

说明：
- 这个脚本是对官方 tools/train.py 的轻量封装，方便把 semimm 的配置快速跑起来。
- 你可以通过 --cfg-options 覆盖配置（例如临时改 MAX_ITERS、路径等）。
"""

import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train DINO + SemiBaseDetector')
    parser.add_argument(
        '--config',
        default='semimm/configs/dino_semibase_mini.py',
        help='训练配置文件路径')
    parser.add_argument('--work-dir', default=None, help='日志与权重输出目录')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='恢复训练。空值表示自动从最新 checkpoint 恢复')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='配置覆盖，格式 key=value，例如 train_cfg.max_iters=50')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='分布式启动方式')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()

    # 与官方脚本保持一致：兼容 torch.distributed 参数
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # 读取配置
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    # 命令行覆盖
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir 优先级：CLI > 配置文件 > 默认
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # resume 逻辑
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # 构建并启动训练
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
