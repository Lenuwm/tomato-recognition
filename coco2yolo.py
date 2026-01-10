# coco2yolo.py
# 用途：把 tomatOD 的 COCO 标注(json) 转为 YOLO txt，并生成 splits: train/val/test
# 适配目录：
#   data/images/train/*.jpg
#   data/images/test/*.jpg
# 输出目录：
#   data/labels/train/*.txt
#   data/labels/test/*.txt
#   data/splits/train.txt  val.txt  test.txt
#
# 关键点：split 文件里写的是 “train/xxx” 这种相对ID（不带后缀），与 dataset.py 逻辑匹配

import os
import json
import random
from collections import defaultdict

def norm_name(s):
    # 关键：如果是list，取第一个元素（或空串）
    if isinstance(s, list):
        s = s[0] if len(s) > 0 else ""
    s = str(s)               # 兜底：强转成字符串
    s = s.strip().lower()
    s = s.replace("_", "-").replace(" ", "")
    return s


# 你的类别顺序（务必与 train.py 里 CFG.classes 一致）
NAME2ID = {
    "unripe": 0,
    "semi-ripe": 1,
    "semiripe": 1,
    "fully-ripe": 2,
    "fullyripe": 2,
    "ripe": 2,  # 兜底
}

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def coco_to_yolo_one(
    coco_json_path: str,
    img_root: str,
    label_root: str,
):
    """
    将一个COCO json 转为 YOLO txt。
    返回：所有样本的“相对id”（例如 train/00001）
    """
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # image_id -> {file_name,width,height}
    id2img = {}
    for img in coco["images"]:
        # COCO里通常自带width/height，优先用它，避免读图
        id2img[img["id"]] = {
            "file_name": img["file_name"],
            "width": img.get("width", None),
            "height": img.get("height", None),
        }

    # category_id -> class_id(0/1/2)
    catid2cls = {}
    unknown = set()
    for c in coco["categories"]:
        nm = norm_name(c["name"])
        if nm not in NAME2ID:
            unknown.add(c["name"])
            continue
        catid2cls[c["id"]] = NAME2ID[nm]

    if unknown:
        print("[Warn] 发现未映射类别名：", list(unknown))
        print("       请检查 tomatOD 的类别命名并补充 NAME2ID 映射。")

    # 收集每张图的标注
    ann_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        if cat_id not in catid2cls:
            continue
        cls = catid2cls[cat_id]
        x, y, w, h = ann["bbox"]  # COCO: top-left + width/height
        ann_by_img[img_id].append((cls, x, y, w, h))

    ids = []
    for img_id, info in id2img.items():
        file_name = info["file_name"]   # 可能是 train/xxx.jpg 或 test/xxx.jpg
        stem_rel = os.path.splitext(file_name)[0]  # 变成 train/xxx
        W, H = info["width"], info["height"]

        # 如果COCO没给width/height，这里就需要你读图补齐；tomatOD一般是有的
        if W is None or H is None:
            # 兜底：尝试读取图片尺寸
            from PIL import Image
            img_path = os.path.join(img_root, file_name)
            with Image.open(img_path) as im:
                W, H = im.size

        # 写label文件（保持 train/xxx.txt 这种结构）
        label_path = os.path.join(label_root, stem_rel + ".txt")
        ensure_dir(os.path.dirname(label_path))

        lines = []
        for cls, x, y, w, h in ann_by_img.get(img_id, []):
            xc = (x + w / 2) / W
            yc = (y + h / 2) / H
            bw = w / W
            bh = h / H
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        # 重要：split里写“train/xxx”这种ID（dataset.py会自动拼 .jpg/.txt）
        ids.append(stem_rel)

    return ids

def write_list(path: str, ids):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(ids) + "\n")

def main():
    # ======= 按你当前目录结构配置（一般不用改） =======
    img_root = "./data/images"
    ann_root = "./data/annotations"
    label_root = "./data/labels"
    split_root = "./data/splits"

    train_json = os.path.join(ann_root, "tomatOD_train.json")
    test_json  = os.path.join(ann_root, "tomatOD_test.json")

    val_ratio = 0.2
    seed = 42

    # ======= 1) 转 train 标注 =======
    train_ids_all = coco_to_yolo_one(train_json, img_root, label_root)

    # 从 train 切出 val
    random.seed(seed)
    random.shuffle(train_ids_all)
    n_val = int(len(train_ids_all) * val_ratio)
    val_ids = train_ids_all[:n_val]
    train_ids = train_ids_all[n_val:]

    write_list(os.path.join(split_root, "train.txt"), train_ids)
    write_list(os.path.join(split_root, "val.txt"), val_ids)

    print(f"[OK] train/val split: train={len(train_ids)}, val={len(val_ids)}")

    # ======= 2) 转 test 标注（可选但建议做） =======
    test_ids = coco_to_yolo_one(test_json, img_root, label_root)
    write_list(os.path.join(split_root, "test.txt"), test_ids)
    print(f"[OK] test split: test={len(test_ids)}")

    print("Done.")

if __name__ == "__main__":
    main()
