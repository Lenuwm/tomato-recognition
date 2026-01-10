# train.py
# 作用：训练入口脚本 + 训练过程记录曲线 + 宏F1/混淆矩阵 + 训练结束test集评估
import os
import time
import csv
import json
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from dataset import TomatoDetDataset, collate_fn
from model import YoloTiny
from loss import build_targets, yolo_loss

# ✅ 统一使用 utils.py 里真实存在的 postprocess，而不是 postprocess_per_class
from utils import decode_preds, postprocess, iou_xyxy

# 随机种子（保证可复现）
torch.manual_seed(0); np.random.seed(0); random.seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class CFG:
    img_dir = "./data/images"
    label_dir = "./data/labels"
    train_list = "./data/splits/train.txt"
    val_list = "./data/splits/val.txt"
    test_list = "./data/splits/test.txt"   # ✅ 3.4：加入 test 划分

    classes = ("unripe", "semi-ripe", "ripe")
    num_classes = 3

    img_size = 640
    S = 20
    A = 3
    anchors = ((12, 14), (26, 30), (64, 72))

    batch_size = 8
    epochs = 30
    lr = 1e-3
    weight_decay = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    conf_th = 0.15
    nms_th = 0.5

    lambda_box = 5.0
    lambda_obj = 1.0
    lambda_cls = 1.0

    out_dir = "./outputs"

    # ✅ 产量估计的平均重量（可替换为你自己的统计值，单位 g）
    w_map = {"unripe": 80.0, "semi-ripe": 120.0, "ripe": 160.0}

cfg = CFG()
os.makedirs(cfg.out_dir, exist_ok=True)


def _gt_to_xyxy_pix(gt: torch.Tensor):
    """GT: (N,5) [cls, xc, yc, w, h] (归一化) -> xyxy(像素) + cls"""
    gt_xyxy, gt_cls = [], []
    for row in gt:
        c = int(row[0].item())
        xc, yc, w, h = row[1], row[2], row[3], row[4]
        x1 = (xc - w / 2) * cfg.img_size
        y1 = (yc - h / 2) * cfg.img_size
        x2 = (xc + w / 2) * cfg.img_size
        y2 = (yc + h / 2) * cfg.img_size
        gt_xyxy.append([x1, y1, x2, y2])
        gt_cls.append(c)

    if len(gt_xyxy) == 0:
        gt_xyxy = torch.zeros((0, 4), device=cfg.device)
        gt_cls = torch.zeros((0,), device=cfg.device, dtype=torch.long)
    else:
        gt_xyxy = torch.tensor(gt_xyxy, device=cfg.device)
        gt_cls = torch.tensor(gt_cls, device=cfg.device, dtype=torch.long)
    return gt_xyxy, gt_cls


def train_one_epoch(model, loader, optimizer, anchors_t):
    model.train()
    meter = {"loss": 0.0, "box": 0.0, "obj": 0.0, "cls": 0.0}
    n = 0
    for imgs, targets, _ in loader:
        imgs = imgs.to(cfg.device)
        pred = model(imgs)  # ✅ 关键：前向

        # ✅ 关键：构造监督张量（anchor匹配/落cell）
        tbox, tobj, tcls = build_targets(targets, anchors_t, cfg.S, cfg.img_size, cfg.num_classes)

        # ✅ 关键：计算损失（box/obj/cls）
        loss, parts = yolo_loss(
            pred, tbox, tobj, tcls,
            anchors_t, cfg.img_size, cfg.S,
            cfg.lambda_box, cfg.lambda_obj, cfg.lambda_cls
        )

        optimizer.zero_grad()
        loss.backward()      # ✅ 关键：反向
        optimizer.step()     # ✅ 关键：更新

        bs = imgs.size(0)
        for k in meter:
            meter[k] += parts.get(k, 0.0) * bs
        n += bs

    for k in meter:
        meter[k] /= max(n, 1)
    return meter


@torch.no_grad()
def infer_batch(model, imgs, anchors_t, agnostic_nms=False):
    """
    agnostic_nms=False：更适合 mAP（按类NMS + 跨类去重）
    agnostic_nms=True ：更适合计数（强去重，避免同一果实多框）
    """
    model.eval()
    pred = model(imgs)

    # ✅ 关键：解码网络输出 -> 候选框 [x1,y1,x2,y2,score,cls]
    dets = decode_preds(pred, anchors_t, cfg.img_size, cfg.conf_th)

    # ✅ 关键：NMS（手写在 utils.py 内部 nms_manual）
    dets = [postprocess(d, cfg.nms_th, agnostic=agnostic_nms) for d in dets]
    return dets


@torch.no_grad()
def evaluate_map_and_count(model, loader, anchors_t, iou_th=0.5):
    """
    输出：
      mAP@0.5（简化AP实现）
      count_MAE / count_RMSE（总数误差）
    """
    model.eval()
    pred_by_class = {c: [] for c in range(cfg.num_classes)}
    gt_count_by_class = {c: 0 for c in range(cfg.num_classes)}
    abs_err, sq_err = [], []

    for imgs, targets, _ in loader:
        imgs = imgs.to(cfg.device)
        dets = infer_batch(model, imgs, anchors_t, agnostic_nms=False)

        for b, det in enumerate(dets):
            gt = targets[b].to(cfg.device)
            gt_xyxy, gt_cls = _gt_to_xyxy_pix(gt)

            # GT 计数（按类）
            for c in gt_cls.tolist():
                gt_count_by_class[int(c)] += 1

            # 计数误差（总数）
            pred_total = int(det.shape[0])
            gt_total = int(gt.shape[0])
            abs_err.append(abs(pred_total - gt_total))
            sq_err.append((pred_total - gt_total) ** 2)

            if det.numel() == 0:
                continue

            # 按类计算 TP/FP 序列（AP）
            for c in range(cfg.num_classes):
                pc = det[det[:, 5].long() == c]
                if pc.numel() == 0:
                    continue
                scores = pc[:, 4]
                boxes = pc[:, 0:4]
                gtc = gt_xyxy[gt_cls == c]

                used = torch.zeros((gtc.shape[0],), dtype=torch.bool, device=cfg.device)
                order = torch.argsort(scores, descending=True)
                for idx in order:
                    box = boxes[idx].unsqueeze(0)
                    sc = scores[idx].item()
                    if gtc.shape[0] == 0:
                        pred_by_class[c].append((sc, 0))
                        continue
                    ious = iou_xyxy(box, gtc).squeeze(0)
                    best = int(torch.argmax(ious).item())
                    if ious[best] >= iou_th and not used[best]:
                        used[best] = True
                        pred_by_class[c].append((sc, 1))
                    else:
                        pred_by_class[c].append((sc, 0))

    def ap_from_scores(items, gt_num):
        if gt_num == 0 or len(items) == 0:
            return 0.0
        items = sorted(items, key=lambda x: x[0], reverse=True)
        tp = fp = 0
        precisions, recalls = [], []
        for _, is_tp in items:
            tp += int(is_tp == 1)
            fp += int(is_tp == 0)
            precisions.append(tp / (tp + fp + 1e-12))
            recalls.append(tp / (gt_num + 1e-12))

        # PR 曲线的插值 + 面积
        mpre = np.maximum.accumulate(np.array(precisions)[::-1])[::-1]
        mrec = np.array(recalls)
        ap, prev_r = 0.0, 0.0
        for p, r in zip(mpre, mrec):
            ap += p * max(r - prev_r, 0.0)
            prev_r = r
        return float(ap)

    aps = [ap_from_scores(pred_by_class[c], gt_count_by_class[c]) for c in range(cfg.num_classes)]
    mAP50 = float(np.mean(aps)) if len(aps) else 0.0
    mae = float(np.mean(abs_err)) if len(abs_err) else 0.0
    rmse = float(np.sqrt(np.mean(sq_err))) if len(sq_err) else 0.0
    return {"mAP@0.5": mAP50, "APs": aps, "count_MAE": mae, "count_RMSE": rmse}


@torch.no_grad()
def evaluate_cls_confmat_macro_f1(model, loader, anchors_t, iou_th=0.5):
    """
    成熟度宏F1 + 混淆矩阵
    规则：只在 IoU>=iou_th 且一对一匹配成功的检测上统计 (gt_cls, pred_cls)
    """
    C = cfg.num_classes
    conf = np.zeros((C, C), dtype=np.int64)
    matched = 0

    for imgs, targets, _ in loader:
        imgs = imgs.to(cfg.device)

        # 用更强去重的 NMS 做分类统计（避免同一果实重复计入）
        dets = infer_batch(model, imgs, anchors_t, agnostic_nms=True)

        for b, det in enumerate(dets):
            gt = targets[b].to(cfg.device)
            if gt.numel() == 0 or det.numel() == 0:
                continue

            gt_xyxy, gt_cls = _gt_to_xyxy_pix(gt)
            pred_xyxy = det[:, 0:4]
            pred_cls = det[:, 5].long()

            if gt_xyxy.shape[0] == 0 or pred_xyxy.shape[0] == 0:
                continue

            ious = iou_xyxy(gt_xyxy, pred_xyxy)  # (Ng, Np)
            best_iou, best_j = ious.max(dim=1)
            order_gt = torch.argsort(best_iou, descending=True)

            used_pred = torch.zeros((pred_xyxy.shape[0],), dtype=torch.bool, device=cfg.device)

            for gi in order_gt:
                j = int(best_j[gi].item())
                if best_iou[gi].item() >= iou_th and (not used_pred[j]):
                    used_pred[j] = True
                    g = int(gt_cls[gi].item())
                    p = int(pred_cls[j].item())
                    conf[g, p] += 1
                    matched += 1

    # 计算 per-class Precision/Recall/F1，再 macro 平均
    eps = 1e-12
    f1s = []
    per_class = {}
    for c in range(C):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = (2 * prec * rec) / (prec + rec + eps)
        per_class[c] = {"precision": float(prec), "recall": float(rec), "f1": float(f1)}
        f1s.append(float(f1))

    macro_f1 = float(np.mean(f1s)) if len(f1s) else 0.0
    return macro_f1, conf, per_class, matched


def save_confusion_matrix(conf: np.ndarray, classes, save_path: str, title: str):
    """把混淆矩阵画出来并保存成 png（用于报告截图）"""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(conf, interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel("Pred")
    ax.set_ylabel("GT")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    # 写数值
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            ax.text(j, i, str(conf[i, j]), ha="center", va="center", fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_history_csv_and_curves(history, exp_dir: str):
    """保存 metrics.csv + 三条曲线图"""
    if len(history) == 0:
        return

    csv_path = os.path.join(exp_dir, "metrics.csv")
    keys = list(history[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)

    epochs = [h["epoch"] for h in history]

    # loss 曲线
    plt.figure()
    plt.plot(epochs, [h["loss"] for h in history])
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "curve_loss.png"), dpi=200)
    plt.close()

    # mAP 曲线
    plt.figure()
    plt.plot(epochs, [h["val_map50"] for h in history])
    plt.xlabel("epoch"); plt.ylabel("mAP@0.5")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "curve_map.png"), dpi=200)
    plt.close()

    # MAE 曲线
    plt.figure()
    plt.plot(epochs, [h["val_mae"] for h in history])
    plt.xlabel("epoch"); plt.ylabel("count MAE")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "curve_mae.png"), dpi=200)
    plt.close()


@torch.no_grad()
def evaluate_yield_mape(model, loader, anchors_t):
    """
    产量估计误差（MAPE）：Yield = Σ count_k * w_k
    GT yield 由 GT计数 * w_k 得到（不需要额外标注重量）
    """
    weights = np.array([cfg.w_map[name] for name in cfg.classes], dtype=np.float32)
    errs = []

    for imgs, targets, _ in loader:
        imgs = imgs.to(cfg.device)
        dets = infer_batch(model, imgs, anchors_t, agnostic_nms=True)

        for b, det in enumerate(dets):
            gt = targets[b].to(cfg.device)
            gt_cls = gt[:, 0].long() if gt.numel() else torch.zeros((0,), device=cfg.device, dtype=torch.long)

            # GT counts
            gt_counts = np.zeros((cfg.num_classes,), dtype=np.float32)
            for c in gt_cls.tolist():
                gt_counts[int(c)] += 1.0

            # Pred counts
            pred_counts = np.zeros((cfg.num_classes,), dtype=np.float32)
            if det.numel():
                for c in det[:, 5].long().tolist():
                    pred_counts[int(c)] += 1.0

            y_gt = float((gt_counts * weights).sum())
            y_pd = float((pred_counts * weights).sum())
            errs.append(abs(y_pd - y_gt) / (y_gt + 1e-6))

    return float(np.mean(errs)) if len(errs) else 0.0


@torch.no_grad()
def benchmark_fps(model, loader, anchors_t, warmup=10, iters=50):
    """简单测速：固定一个 batch，统计端到端（forward+decode+NMS）FPS"""
    model.eval()
    it = iter(loader)
    imgs, _, _ = next(it)
    imgs = imgs.to(cfg.device)

    # warmup
    for _ in range(warmup):
        _ = infer_batch(model, imgs, anchors_t, agnostic_nms=True)

    if cfg.device.startswith("cuda"):
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    n = 0
    for _ in range(iters):
        _ = infer_batch(model, imgs, anchors_t, agnostic_nms=True)
        n += imgs.size(0)

    if cfg.device.startswith("cuda"):
        torch.cuda.synchronize()

    dt = time.perf_counter() - t0
    fps = n / max(dt, 1e-9)
    return float(fps)


from datetime import datetime
def main():
    # === 每次训练一个新目录，避免覆盖 ===
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    anchor_tag = "_".join([f"{w}x{h}" for (w, h) in cfg.anchors])
    exp_dir = os.path.join(cfg.out_dir, f"exp_{run_id}_{anchor_tag}")
    os.makedirs(exp_dir, exist_ok=True)
    print("[ExpDir]", exp_dir)

    best_path = os.path.join(exp_dir, "best.pt")
    last_path = os.path.join(exp_dir, "last.pt")

    anchors_t = torch.tensor(cfg.anchors, device=cfg.device)

    train_set = TomatoDetDataset(cfg.train_list, cfg.img_dir, cfg.label_dir, cfg.img_size, augment=True)
    val_set = TomatoDetDataset(cfg.val_list, cfg.img_dir, cfg.label_dir, cfg.img_size, augment=False)
    test_set = TomatoDetDataset(cfg.test_list, cfg.img_dir, cfg.label_dir, cfg.img_size, augment=False)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,  num_workers=2, collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=cfg.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
    test_loader  = DataLoader(test_set,  batch_size=cfg.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    model = YoloTiny(cfg.num_classes, cfg.S, cfg.A).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_map = -1.0
    history = []  # 每个epoch记录一次

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        meter = train_one_epoch(model, train_loader, optimizer, anchors_t)
        val_metrics = evaluate_map_and_count(model, val_loader, anchors_t)

        # 宏F1 + 混淆矩阵（这里每轮算 macroF1；混淆矩阵只在 best 时保存）
        val_f1, val_conf, _, matched = evaluate_cls_confmat_macro_f1(model, val_loader, anchors_t, iou_th=0.5)

        dt = time.time() - t0
        print(f"[Epoch {epoch:03d}] "
              f"loss={meter['loss']:.4f} (box={meter['box']:.4f}, obj={meter['obj']:.4f}, cls={meter['cls']:.4f}) "
              f"| val mAP@0.5={val_metrics['mAP@0.5']:.4f} "
              f"| val countMAE={val_metrics['count_MAE']:.3f} "
              f"| val macroF1={val_f1:.4f} (matched={matched}) "
              f"| time={dt:.1f}s")

        # 每轮存 last
        torch.save({"model": model.state_dict()}, last_path)

        # 记录 history
        history.append({
            "epoch": epoch,
            "loss": float(meter["loss"]),
            "box": float(meter["box"]),
            "obj": float(meter["obj"]),
            "cls": float(meter["cls"]),
            "val_map50": float(val_metrics["mAP@0.5"]),
            "val_mae": float(val_metrics["count_MAE"]),
            "val_rmse": float(val_metrics["count_RMSE"]),
            "val_macroF1": float(val_f1),
        })

        # best 保存 + 保存最佳轮的混淆矩阵图
        if val_metrics["mAP@0.5"] > best_map:
            best_map = val_metrics["mAP@0.5"]
            torch.save({"model": model.state_dict()}, best_path)
            print(f"  [Save] best checkpoint -> {best_path}")

            save_confusion_matrix(
                val_conf, cfg.classes,
                os.path.join(exp_dir, "confusion_val_best.png"),
                title=f"Val Confusion (best mAP) @ epoch {epoch}"
            )

    # 训练完成后保存 CSV + 曲线图
    save_history_csv_and_curves(history, exp_dir)
    print("[Saved] metrics.csv & curves in:", exp_dir)

    # 用 best.pt 在 test 上跑最终指标并保存
    ckpt = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_det = evaluate_map_and_count(model, test_loader, anchors_t)
    test_f1, test_conf, _, test_matched = evaluate_cls_confmat_macro_f1(model, test_loader, anchors_t, iou_th=0.5)
    test_yield_mape = evaluate_yield_mape(model, test_loader, anchors_t)
    test_fps = benchmark_fps(model, test_loader, anchors_t)

    save_confusion_matrix(
        test_conf, cfg.classes,
        os.path.join(exp_dir, "confusion_test.png"),
        title=f"Test Confusion @ best model (matched={test_matched})"
    )

    test_summary = {
        "mAP@0.5": float(test_det["mAP@0.5"]),
        "count_MAE": float(test_det["count_MAE"]),
        "count_RMSE": float(test_det["count_RMSE"]),
        "macroF1": float(test_f1),
        "yield_MAPE": float(test_yield_mape),
        "FPS(batch_fixed)": float(test_fps),
        "anchors": cfg.anchors,
        "conf_th": cfg.conf_th,
        "nms_th": cfg.nms_th,
        "img_size": cfg.img_size,
    }

    with open(os.path.join(exp_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_summary, f, ensure_ascii=False, indent=2)

    print("[Test Summary]", test_summary)
    print("Training done.")


if __name__ == "__main__":
    main()
