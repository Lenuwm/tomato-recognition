# utils.py
# 作用：提供与模型/训练无关的通用功能
# - 图像letterbox与坐标变换
# - IoU计算（核心计算：交并集）
# - 手写NMS（核心计算：排序+逐个抑制）
# - 预测解码decode（核心计算：从网络输出到像素bbox）
# - 计数与产量估计、可视化保存

import os
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

# Pillow v9.1+ exposes resampling filters under Image.Resampling; older versions use Image.<FILTER>
try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR

def letterbox(img: Image.Image, new_size: int):
    """按比例缩放并填充到new_size，返回(新图, 缩放比例r, pad_dx, pad_dy)"""
    w, h = img.size
    r = min(new_size / w, new_size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    img_resized = img.resize((nw, nh), RESAMPLE_BILINEAR)

    canvas = Image.new("RGB", (new_size, new_size), (114, 114, 114))
    dx = (new_size - nw) // 2
    dy = (new_size - nh) // 2
    canvas.paste(img_resized, (dx, dy))
    return canvas, r, dx, dy

def yolo_norm_to_xyxy(norm_box, img_w, img_h):
    """YOLO格式(cls, xc, yc, w, h) -> 像素xyxy"""
    _, xc, yc, bw, bh = norm_box
    x1 = (xc - bw / 2) * img_w
    y1 = (yc - bh / 2) * img_h
    x2 = (xc + bw / 2) * img_w
    y2 = (yc + bh / 2) * img_h
    return x1, y1, x2, y2

def iou_xyxy(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    计算IoU：box1(N,4), box2(M,4) -> IoU(N,M)
    box格式: (x1,y1,x2,y2)
    """
    tl = torch.max(box1[:, None, :2], box2[None, :, :2])  # 交集左上角 = max
    br = torch.min(box1[:, None, 2:], box2[None, :, 2:])  # 交集右下角 = min

    wh = (br - tl).clamp(min=0)                           # 关键：交集宽高<0置0
    inter = wh[:, :, 0] * wh[:, :, 1]                     # 交集面积

    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)

    union = area1[:, None] + area2[None, :] - inter       # 关键：并集
    return inter / (union + 1e-9)                          # 关键：除零保护

def nms_manual(boxes: torch.Tensor, scores: torch.Tensor, iou_th: float):
    """
    手写NMS
    1) 过滤无效框（w/h<=0）
    2) 按scores降序排序
    3) 逐个保留 + IoU抑制
    返回：LongTensor 索引（在CPU上也可用，在GPU上也可用）
    """
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    # 1) 过滤无效框（避免 x2<x1 或 y2<y1 导致IoU异常）
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    valid = (x2 > x1) & (y2 > y1) & torch.isfinite(scores)
    if valid.sum() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    boxes_v = boxes[valid]
    scores_v = scores[valid]
    idx_map = torch.nonzero(valid).squeeze(1)  # 记录原始索引

    # 2) 排序
    order = torch.argsort(scores_v, descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]                 # 注意：这里 i 是“有效框集合中的索引”，不是 item()
        keep.append(i)

        if order.numel() == 1:
            break

        rest = order[1:]
        iou = iou_xyxy(boxes_v[i].unsqueeze(0), boxes_v[rest]).squeeze(0)
        order = rest[iou <= iou_th]

    keep = torch.stack(keep)         # (K,)
    return idx_map[keep]             # 映射回原始 boxes 的索引


def decode_preds(pred: torch.Tensor, anchors: torch.Tensor, img_size: int, conf_th: float):
    """
    pred: (B,S,S,A,5+C)
    输出每张图候选框 (N,6): [x1,y1,x2,y2,score,cls]
    """
    B, S, _, A, D = pred.shape
    C = D - 5
    device = pred.device

    tx, ty, tw, th = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]
    to = pred[..., 4]
    tcls = pred[..., 5:]  # (B,S,S,A,C)

    gx = torch.sigmoid(tx)
    gy = torch.sigmoid(ty)

    grid_y, grid_x = torch.meshgrid(torch.arange(S, device=device),
                                    torch.arange(S, device=device), indexing="ij")
    grid_x = grid_x[None, :, :, None].float()  # (1,S,S,1)
    grid_y = grid_y[None, :, :, None].float()

    # 关键：中心点解码
    x = (gx + grid_x) / S
    y = (gy + grid_y) / S

    # 关键：宽高解码（anchor * exp）
    aw = anchors[:, 0].view(1, 1, 1, A).float() / img_size
    ah = anchors[:, 1].view(1, 1, 1, A).float() / img_size
    w = aw * torch.exp(tw).clamp(max=10)
    h = ah * torch.exp(th).clamp(max=10)

    x1 = (x - w / 2) * img_size
    y1 = (y - h / 2) * img_size
    x2 = (x + w / 2) * img_size
    y2 = (y + h / 2) * img_size

    obj = torch.sigmoid(to)
    cls_prob = torch.softmax(tcls, dim=-1)  # 建议：多类用 softmax 更合理
    cls_score, cls_id = torch.max(cls_prob, dim=-1)


    score = obj * cls_score                                 # 关键：置信度融合


    outs = []
    topk = 300  # 100~300都行

    for b in range(B):
        sb = score[b].reshape(-1)  # (S*S*A,)
        if sb.numel() == 0:
            outs.append(torch.zeros((0, 6), device=device))
            continue

        # 先按 conf_th 过滤
        keep_mask = sb >= conf_th
        idx = torch.nonzero(keep_mask).squeeze(1)
        if idx.numel() == 0:
            outs.append(torch.zeros((0, 6), device=device))
            continue

        # 再做 topk，避免候选过多
        if idx.numel() > topk:
            vals = sb[idx]
            top_idx = torch.topk(vals, k=topk).indices
            idx = idx[top_idx]

        # 把一维 idx 映射回 (y,x,a)
        # idx = (y*S + x)*A + a
        a = idx % A
        tmp = idx // A
        xg = tmp % S
        yg = tmp // S

        out = torch.stack([
            x1[b, yg, xg, a], y1[b, yg, xg, a], x2[b, yg, xg, a], y2[b, yg, xg, a],
            sb[idx], cls_id[b, yg, xg, a].float()
        ], dim=1)
        outs.append(out)
    return outs

# def postprocess_per_class(det: torch.Tensor, nms_th: float):
#     """按类别分别做NMS，避免不同类互相抑制"""
#     if det.numel() == 0:
#         return det
#     out = []
#     cls_ids = det[:, 5].long()
#     for c in torch.unique(cls_ids):
#         m = cls_ids == c
#         boxes = det[m][:, 0:4]
#         scores = det[m][:, 4]
#         keep = nms_manual(boxes, scores, nms_th)            # 关键：手写NMS
#         out.append(det[m][keep])
#     return torch.cat(out, dim=0) if len(out) else det

def postprocess(det: torch.Tensor, nms_th: float, agnostic: bool = True, cross_class_th: float = 0.7):
    """
    agnostic=True: 直接 class-agnostic NMS（推荐用于计数）
    agnostic=False: 先 per-class NMS，再用 cross_class_th 做一次跨类轻量去重
    """
    if det.numel() == 0:
        return det

    if agnostic:
        boxes = det[:, 0:4]
        scores = det[:, 4]
        keep = nms_manual(boxes, scores, nms_th)
        return det[keep]

    # 1) 先按类别分别 NMS（你原来的逻辑）
    out = []
    cls_ids = det[:, 5].long()
    for c in torch.unique(cls_ids):
        m = cls_ids == c
        boxes = det[m][:, 0:4]
        scores = det[m][:, 4]
        keep = nms_manual(boxes, scores, nms_th)
        out.append(det[m][keep])
    det2 = torch.cat(out, dim=0) if len(out) else det

    # 2) 再做一次“跨类别去重”：阈值设大一点，只删除几乎重合的重复框
    boxes2 = det2[:, 0:4]
    scores2 = det2[:, 4]
    keep2 = nms_manual(boxes2, scores2, cross_class_th)
    return det2[keep2]

def count_and_yield(det: torch.Tensor, classes, w_map: dict):
    """按类别计数 + 产量估计 Yield=Σ count_k*w_k"""
    counts = {name: 0 for name in classes}
    if det.numel() != 0:
        cls = det[:, 5].long().cpu().numpy()
        for c in cls:
            counts[classes[int(c)]] += 1
    counts["total"] = sum(counts[k] for k in classes)
    y = sum(counts[k] * w_map[k] for k in classes)
    return counts, float(y)

def draw_boxes(img: Image.Image, det: torch.Tensor, classes, save_path: str):
    """把bbox+类别+置信度画到图上并保存"""
    draw = ImageDraw.Draw(img)
    for row in det.cpu().numpy():
        x1, y1, x2, y2, s, c = row
        c = int(c)
        name = classes[c]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        draw.text((x1 + 2, y1 + 2), f"{name}:{s:.2f}", fill=(255, 255, 0))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)
