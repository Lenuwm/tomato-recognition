# fix_splits.py
import os

def detect_prefix(img_root, stem):
    for sub in ["train", "test"]:
        for ext in [".jpg", ".jpeg", ".png"]:
            p = os.path.join(img_root, sub, stem + ext)
            if os.path.exists(p):
                return f"{sub}/{stem}"
    return None

def fix_one(split_path, img_root):
    new_lines = []
    bad = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # 已经带前缀就跳过
            if s.startswith("train/") or s.startswith("test/"):
                new_lines.append(s)
                continue
            prefix = detect_prefix(img_root, s)
            if prefix is None:
                bad.append(s)
            else:
                new_lines.append(prefix)

    with open(split_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")

    print(f"[OK] fixed {os.path.basename(split_path)} -> {len(new_lines)} lines")
    if bad:
        print("[WARN] not found images for these ids (show first 10):")
        print("\n".join(bad[:10]))

if __name__ == "__main__":
    img_root = "./data/images"
    for sp in ["train.txt", "val.txt", "test.txt"]:
        p = os.path.join("./data/splits", sp)
        if os.path.exists(p):
            fix_one(p, img_root)
