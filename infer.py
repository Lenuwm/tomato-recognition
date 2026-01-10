# infer.py（建议改造版关键片段）
import os, argparse
import numpy as np
import torch
from typing import Optional
from PIL import Image
from model import YoloTiny
from utils import letterbox, decode_preds, postprocess, count_and_yield, draw_boxes  # 用 agnostic 的 postprocess 更适合计数

class CFG:
    classes = ("unripe", "semi-ripe", "ripe")
    num_classes = 3
    img_size = 640
    S = 20
    A = 3
    anchors = ((12, 14), (26, 30), (64, 72))
    conf_th = 0.15
    nms_th = 0.5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "./outputs/yolo_like_tomato.pt"
    out_dir = "./outputs"
    w_map = {"unripe": 80.0, "semi-ripe": 120.0, "ripe": 160.0}
cfg = CFG()

def load_model(ckpt_path: str):
    model = YoloTiny(cfg.num_classes, cfg.S, cfg.A).to(cfg.device)
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    # 兼容两种保存方式：{"model": state_dict} 或 直接 state_dict
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model

@torch.no_grad()
def infer_one(model, image_path: str, save_path: Optional[str] = None):
    anchors_t = torch.tensor(cfg.anchors, device=cfg.device)

    img0 = Image.open(image_path).convert("RGB")
    img_lb, r, dx, dy = letterbox(img0, cfg.img_size)

    x = torch.from_numpy(np.array(img_lb)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    x = x.to(cfg.device)

    pred = model(x)
    det = decode_preds(pred, anchors_t, cfg.img_size, cfg.conf_th)[0]
    det = postprocess(det, cfg.nms_th, agnostic=True)  # 推荐：不同类别也抑制，减少重复框（计数更稳）

    counts, y = count_and_yield(det, cfg.classes, cfg.w_map)
    print("Counts:", counts, "Yield(g):", y)

    if save_path is None:
        save_path = os.path.join(cfg.out_dir, "infer_vis.jpg")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    draw_boxes(img_lb, det, cfg.classes, save_path)
    print("Saved:", save_path)

def infer_from_test_list(test_list="./data/splits/test.txt", save_dir="./outputs/vis_test", max_n=30):
    os.makedirs(save_dir, exist_ok=True)
    with open(test_list, "r", encoding="utf-8") as f:
        ids = [x.strip() for x in f.readlines() if x.strip()][:max_n]

    model = load_model(cfg.ckpt_path)

    for img_id in ids:
        base = os.path.join("./data/images", img_id)
        image_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            p = base + ext
            if os.path.exists(p):
                image_path = p
                break
        if image_path is None:
            print("[Skip] not found:", base)
            continue

        out_name = os.path.basename(base) + f"_conf{cfg.conf_th:.2f}.jpg"
        infer_one(model, image_path, save_path=os.path.join(save_dir, out_name))

if __name__ == "__main__":
    infer_from_test_list(max_n=30)
