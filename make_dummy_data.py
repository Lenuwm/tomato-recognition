# make_dummy_data.py
import os, random
import numpy as np
from PIL import Image, ImageDraw

ROOT = "."
IMG_DIR = os.path.join(ROOT, "data", "images")
LAB_DIR = os.path.join(ROOT, "data", "labels")
SPL_DIR = os.path.join(ROOT, "data", "splits")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LAB_DIR, exist_ok=True)
os.makedirs(SPL_DIR, exist_ok=True)

def gen_one(img_size=640, max_objs=8):
    img = Image.new("RGB", (img_size, img_size), (30, 30, 30))
    draw = ImageDraw.Draw(img)

    n = random.randint(1, max_objs)
    labels = []
    for _ in range(n):
        cls = random.randint(0, 2)  # 0/1/2 对应 unripe/semi-ripe/ripe
        # 随机生成一个bbox（像素）
        w = random.randint(30, 140)
        h = random.randint(30, 140)
        x1 = random.randint(0, img_size - w - 1)
        y1 = random.randint(0, img_size - h - 1)
        x2, y2 = x1 + w, y1 + h

        # 画个“番茄样子”的椭圆，便于你看可视化（纯模拟）
        color = [(60,180,60), (220,160,60), (220,60,60)][cls]
        draw.ellipse([x1, y1, x2, y2], fill=color, outline=(255,255,255))

        # YOLO归一化标签：cls xc yc bw bh
        xc = ((x1 + x2) / 2) / img_size
        yc = ((y1 + y2) / 2) / img_size
        bw = (x2 - x1) / img_size
        bh = (y2 - y1) / img_size
        labels.append((cls, xc, yc, bw, bh))
    return img, labels

def main(N=120, img_size=640, split=0.8):
    ids = []
    for i in range(N):
        img_id = f"{i:05d}"
        img, labs = gen_one(img_size=img_size)
        img.save(os.path.join(IMG_DIR, img_id + ".jpg"), quality=95)

        with open(os.path.join(LAB_DIR, img_id + ".txt"), "w", encoding="utf-8") as f:
            for cls, xc, yc, bw, bh in labs:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        ids.append(img_id)

    random.shuffle(ids)
    n_train = int(len(ids) * split)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]

    with open(os.path.join(SPL_DIR, "train.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_ids) + "\n")
    with open(os.path.join(SPL_DIR, "val.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_ids) + "\n")

    print("Dummy data generated.")
    print("train:", len(train_ids), "val:", len(val_ids))

if __name__ == "__main__":
    main()
