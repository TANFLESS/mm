#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从标准 COCO 训练集抽取一个 mini 子集，便于快速测试半监督训练流程。

功能：
1) 抽取 N 张图片（默认 100）到目标目录；
2) 生成 labeled 标注（保留这部分图片的真值框）；
3) 生成 unlabeled 标注（仅保留图片和类别，annotations 为空）；
4) 生成 val 标注（默认用抽样全集做验证，方便冒烟测试）。

注意：
- 这是“工程自测”数据，不是正式实验切分。
- 正式实验请使用官方 semi split。
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List


def load_json(path: Path) -> Dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)


def pick_images(images: List[Dict], num_images: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    if num_images >= len(images):
        return images
    return rng.sample(images, num_images)


def split_labeled_unlabeled(images: List[Dict], labeled_ratio: float, seed: int):
    rng = random.Random(seed)
    ids = [img['id'] for img in images]
    rng.shuffle(ids)
    labeled_num = max(1, int(len(ids) * labeled_ratio))
    labeled_ids = set(ids[:labeled_num])
    unlabeled_ids = set(ids[labeled_num:])
    if not unlabeled_ids:
        # 保证至少有一张无标注图，便于半监督流程可运行
        one = next(iter(labeled_ids))
        labeled_ids.remove(one)
        unlabeled_ids.add(one)
    return labeled_ids, unlabeled_ids


def filter_annotations(annotations: List[Dict], image_id_set: set) -> List[Dict]:
    return [ann for ann in annotations if ann['image_id'] in image_id_set]


def filter_images(images: List[Dict], image_id_set: set) -> List[Dict]:
    return [img for img in images if img['id'] in image_id_set]


def copy_images(images: List[Dict], src_img_dir: Path, dst_img_dir: Path) -> None:
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        src = src_img_dir / img['file_name']
        dst = dst_img_dir / img['file_name']
        if not src.exists():
            raise FileNotFoundError(f'找不到源图片: {src}')
        if not dst.exists():
            shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description='生成 mini COCO（用于半监督冒烟测试）')
    parser.add_argument('--src-coco-root', required=True, type=Path,
                        help='原始 COCO 根目录（含 train2017 与 annotations）')
    parser.add_argument('--dst-root', required=True, type=Path,
                        help='输出 mini COCO 根目录')
    parser.add_argument('--num-images', type=int, default=100,
                        help='抽样图片数量')
    parser.add_argument('--labeled-ratio', type=float, default=0.5,
                        help='labeled 图像占比，剩余作为 unlabeled')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    if not (0.0 < args.labeled_ratio < 1.0):
        raise ValueError('--labeled-ratio 必须在 (0,1) 区间内')

    src_train_json = args.src_coco_root / 'annotations' / 'instances_train2017.json'
    src_train_img_dir = args.src_coco_root / 'train2017'

    coco = load_json(src_train_json)
    images = coco['images']
    annotations = coco['annotations']
    categories = coco['categories']

    sampled_images = pick_images(images, args.num_images, args.seed)
    sampled_ids = {img['id'] for img in sampled_images}
    sampled_annotations = filter_annotations(annotations, sampled_ids)

    labeled_ids, unlabeled_ids = split_labeled_unlabeled(sampled_images, args.labeled_ratio, args.seed)

    labeled_images = filter_images(sampled_images, labeled_ids)
    unlabeled_images = filter_images(sampled_images, unlabeled_ids)

    labeled_annotations = filter_annotations(sampled_annotations, labeled_ids)

    # 1) 复制图片
    dst_img_dir = args.dst_root / 'train2017'
    copy_images(sampled_images, src_train_img_dir, dst_img_dir)

    # 2) 生成标注
    ann_dir = args.dst_root / 'annotations'

    # 全量 mini（可用于调试）
    mini_all = {
        'images': sampled_images,
        'annotations': sampled_annotations,
        'categories': categories,
    }
    save_json(mini_all, ann_dir / 'instances_train2017_mini.json')

    # labeled：保留框标注
    mini_labeled = {
        'images': labeled_images,
        'annotations': labeled_annotations,
        'categories': categories,
    }
    save_json(mini_labeled, ann_dir / 'instances_train2017_mini_labeled.json')

    # unlabeled：只保留图片与类别，annotations 置空
    mini_unlabeled = {
        'images': unlabeled_images,
        'annotations': [],
        'categories': categories,
    }
    save_json(mini_unlabeled, ann_dir / 'instances_train2017_mini_unlabeled.json')

    # val：为了冒烟方便，先复用 mini 全量
    save_json(mini_all, ann_dir / 'instances_val2017_mini.json')

    print('✅ mini COCO 生成完成')
    print(f'输出目录: {args.dst_root}')
    print(f'总图片数: {len(sampled_images)}')
    print(f'labeled 图片数: {len(labeled_images)} / anns: {len(labeled_annotations)}')
    print(f'unlabeled 图片数: {len(unlabeled_images)} / anns: 0')


if __name__ == '__main__':
    main()
