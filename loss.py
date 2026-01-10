# loss.py
# 作用：训练侧核心计算
# - build_targets：GT分配到(cell, best_anchor)
# - wh_iou：用于挑选最佳anchor（只比较w/h）
# - yolo_loss：定位(IoU loss)+置信度(BCE)+分类(BCE) 关键步骤手写

import torch
import torch.nn.functional as F
from utils import iou_xyxy

def wh_iou(gt_wh: torch.Tensor, anchor_wh: torch.Tensor) -> torch.Tensor:
    """仅比较宽高的IoU，用于anchor匹配"""
    gt = gt_wh[:, None, :]          # (N,1,2)
    an = anchor_wh[None, :, :]      # (1,A,2)
    inter = torch.min(gt[..., 0], an[..., 0]) * torch.min(gt[..., 1], an[..., 1])
    union = gt[..., 0]*gt[..., 1] + an[..., 0]*an[..., 1] - inter
    return inter / (union + 1e-9)

def build_targets(targets, anchors: torch.Tensor, S: int, img_size: int, num_classes: int):
    """
    targets: list[tensor(N,5)] each row [cls, xc, yc, w, h] normalized
    输出监督张量：
    - tbox (B,S,S,A,4)
    - tobj (B,S,S,A)
    - tcls (B,S,S,A,C)
    """
    B = len(targets)
    A = anchors.shape[0]
    device = anchors.device

    tbox = torch.zeros((B, S, S, A, 4), device=device)
    tobj = torch.zeros((B, S, S, A), device=device)
    tcls = torch.zeros((B, S, S, A, num_classes), device=device)

    for b in range(B):
        if targets[b].numel() == 0:
            continue
        gt = targets[b].to(device)
        cls = gt[:, 0].long()
        xc, yc, w, h = gt[:, 1], gt[:, 2], gt[:, 3], gt[:, 4]

        # 关键：GT所属cell
        gx = (xc * S).clamp(0, S - 1e-3)
        gy = (yc * S).clamp(0, S - 1e-3)
        cx = gx.long()
        cy = gy.long()

        # 关键：选择最佳anchor（gt_wh与anchor_wh的IoU最大）
        gt_wh_pix = torch.stack([w * img_size, h * img_size], dim=1)
        ious = wh_iou(gt_wh_pix, anchors)              # (N,A)
        topk = 2
        top_a = torch.topk(ious, k=min(topk, ious.shape[1]), dim=1).indices  # (N,topk)

        for i in range(gt.shape[0]):
            xci, yci = cx[i].item(), cy[i].item()
            for kk in range(top_a.shape[1]):
                a = top_a[i, kk].item()
                tobj[b, int(yci), int(xci), int(a)] = 1.0
                tbox[b, int(yci), int(xci), int(a)] = torch.stack([xc[i], yc[i], w[i], h[i]])
                tcls[b, int(yci), int(xci), int(a), int(cls[i].item())] = 1.0

    return tbox, tobj, tcls

def yolo_loss(pred, tbox, tobj, tcls, anchors, img_size, S, lambda_box, lambda_obj, lambda_cls):
    """
    pred: (B,S,S,A,5+C) raw logits
    计算简化YOLO loss
    """
    p_obj = pred[..., 4]
    obj_loss = F.binary_cross_entropy_with_logits(p_obj, tobj, reduction="mean")  # 关键：置信度BCE

    pos_mask = tobj > 0.5
    if pos_mask.sum() == 0:
        total = lambda_obj * obj_loss
        return total, {"loss": total.item(), "box": 0.0, "obj": obj_loss.item(), "cls": 0.0}

    p_pos = pred[pos_mask]          # (Npos, 5+C)
    t_pos = tbox[pos_mask]          # (Npos, 4)
    cls_pos = tcls[pos_mask]        # (Npos, C)

    # 为解码定位loss，需要pos的(cell,anchor)索引
    pos_idx = pos_mask.nonzero(as_tuple=False)  # [b,y,x,a]
    cy = pos_idx[:, 1].float()
    cx = pos_idx[:, 2].float()
    a_id = pos_idx[:, 3]

    tx, ty, tw, th = p_pos[:, 0], p_pos[:, 1], p_pos[:, 2], p_pos[:, 3]
    gx = torch.sigmoid(tx)
    gy = torch.sigmoid(ty)

    # 关键：中心点解码
    x = (gx + cx) / S
    y = (gy + cy) / S

    # 关键：宽高解码
    aw = anchors[a_id, 0].float() / img_size
    ah = anchors[a_id, 1].float() / img_size
    w = aw * torch.exp(tw).clamp(max=10)
    h = ah * torch.exp(th).clamp(max=10)

    # 转xyxy像素
    px1 = (x - w/2) * img_size
    py1 = (y - h/2) * img_size
    px2 = (x + w/2) * img_size
    py2 = (y + h/2) * img_size
    p_xyxy = torch.stack([px1, py1, px2, py2], dim=1)

    # GT转xyxy像素
    txc, tyc, tw_gt, th_gt = t_pos[:, 0], t_pos[:, 1], t_pos[:, 2], t_pos[:, 3]
    gx1 = (txc - tw_gt/2) * img_size
    gy1 = (tyc - th_gt/2) * img_size
    gx2 = (txc + tw_gt/2) * img_size
    gy2 = (tyc + th_gt/2) * img_size
    g_xyxy = torch.stack([gx1, gy1, gx2, gy2], dim=1)

    # 关键：定位loss = 1 - IoU
    iou = iou_xyxy(p_xyxy, g_xyxy).diag()
    box_loss = (1.0 - iou).mean()

    # 关键：分类loss（正样本位置）
    target_cls = torch.argmax(cls_pos, dim=1)  # (Npos,) 由one-hot转类别id
    cls_loss = F.cross_entropy(p_pos[:, 5:], target_cls, reduction="mean")

    total = lambda_box * box_loss + lambda_obj * obj_loss + lambda_cls * cls_loss
    return total, {"loss": total.item(), "box": box_loss.item(), "obj": obj_loss.item(), "cls": cls_loss.item()}
