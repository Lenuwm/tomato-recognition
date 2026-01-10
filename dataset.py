# dataset.py
# 作用：数据读取与预处理
# - 读取图像与YOLO格式标注(txt)
# - letterbox到固定输入尺寸，并同步变换标注坐标
# - 可选数据增强（flip等）
# - DataLoader的collate_fn（处理变长GT）

import os
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from utils import letterbox, yolo_norm_to_xyxy


def find_image(img_dir, img_id):
    base = os.path.join(img_dir, img_id)
    for ext in [".jpg", ".jpeg", ".png"]:
        p = base + ext
        if os.path.exists(p):
            return p
    return None

class TomatoDetDataset(Dataset):
    def __init__(self, list_file: str, img_dir: str, label_dir: str, img_size: int, augment: bool = False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment

        with open(list_file, "r", encoding="utf-8") as f:
            self.ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.ids)

    def _read_labels(self, lab_path: str) -> np.ndarray:
        if not os.path.exists(lab_path):
            return np.zeros((0, 5), dtype=np.float32)
        rows = []
        with open(lab_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, w, h = map(float, parts)
                rows.append([cls, xc, yc, w, h])
        return np.array(rows, dtype=np.float32)


    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = find_image(self.img_dir, img_id)
        if img_path is None:
            raise FileNotFoundError(f"Image not found for id={img_id} in {self.img_dir}")
        lab_path = os.path.join(self.label_dir, img_id + ".txt")
        if not os.path.exists(lab_path):
            lab_path2 = os.path.join(self.label_dir, os.path.basename(img_id) + ".txt")
            if os.path.exists(lab_path2):
                lab_path = lab_path2

        img = Image.open(img_path).convert("RGB")
        labels = self._read_labels(lab_path)  # (N,5) YOLO norm on original image

        # 1) letterbox
        img_lb, r, dx, dy = letterbox(img, self.img_size)

        # 2) 同步变换标注（关键：先norm->原图像素->缩放+平移->再归一化）
        ow, oh = img.size
        new_labels = []
        for lab in labels:
            cls = lab[0]
            x1, y1, x2, y2 = yolo_norm_to_xyxy(lab, ow, oh)
            x1, y1, x2, y2 = x1 * r + dx, y1 * r + dy, x2 * r + dx, y2 * r + dy

            xc = (x1 + x2) / 2 / self.img_size
            yc = (y1 + y2) / 2 / self.img_size
            bw = (x2 - x1) / self.img_size
            bh = (y2 - y1) / self.img_size
            new_labels.append([cls, xc, yc, bw, bh])

        new_labels = np.array(new_labels, dtype=np.float32) if len(new_labels) else np.zeros((0, 5), dtype=np.float32)

        # 3) 简化增强：随机水平翻转
        if self.augment and np.random.rand() < 0.5:
            img_lb = ImageOps.mirror(img_lb)
            if len(new_labels):
                new_labels[:, 1] = 1.0 - new_labels[:, 1]  # 关键：flip后xc=1-xc

        # 4) 转tensor
        img_t = torch.from_numpy(np.array(img_lb)).permute(2, 0, 1).float() / 255.0
        targets = torch.from_numpy(new_labels)
        return img_t, targets, img_id

def collate_fn(batch):
    imgs, targets, ids = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, list(targets), list(ids)
