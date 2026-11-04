import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

MIN_VISIBLE = 5   # drop animals with fewer visible keypoints than this

# ---------- 1. Load ----------
with open("data/keypoints.json") as f:
    data = json.load(f)
id2name = {c["id"]: c["name"] for c in data["categories"]}

rows = []
for a in data["annotations"]:
    kp = np.array(a["keypoints"], dtype=np.float32)   # (20, 3)
    rows.append({
        "image_id": a["image_id"],
        "species": id2name[a["category_id"]],
        "bbox": a["bbox"],
        "kpts": kp[:, :2],
        "vis": kp[:, 2],
    })
df = pd.DataFrame(rows)
df["n_visible"] = df["vis"].apply(lambda v: int((v > 0).sum()))

# ---------- 2. Clean ----------
before = len(df)
df = df[df["n_visible"] >= MIN_VISIBLE].reset_index(drop=True)
print(f"Kept {len(df)} of {before} animals")
print(df["species"].value_counts())

# ---------- 3. Detect bbox format ----------
def inside_fraction_xyxy():
    inside = total = 0
    for bbox, kpts, vis in zip(df["bbox"], df["kpts"], df["vis"]):
        x0, y0, x1, y1 = bbox
        p = kpts[vis > 0]
        inside += ((p[:, 0] >= x0) & (p[:, 0] <= x1) &
                   (p[:, 1] >= y0) & (p[:, 1] <= y1)).sum()
        total += len(p)
    return inside / total

BBOX_FMT = "xyxy" if inside_fraction_xyxy() > 0.95 else "xywh"
print("Detected bbox format:", BBOX_FMT)

# ---------- 4. Normalize ----------
def normalize(bbox, kpts, vis):
    x0, y0, a, b = bbox
    x1, y1 = (a, b) if BBOX_FMT == "xyxy" else (x0 + a, y0 + b)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    scale = max(x1 - x0, y1 - y0, 1e-6)
    v = vis > 0
    out = np.zeros_like(kpts)
    out[v, 0] = (kpts[v, 0] - cx) / scale
    out[v, 1] = (kpts[v, 1] - cy) / scale
    # features: 40 normalized coordinates + 20 visibility flags
    return np.concatenate([out.flatten(), v.astype(np.float32)])

X = np.stack([normalize(b, k, v) for b, k, v in zip(df["bbox"], df["kpts"], df["vis"])])
y = df["species"].to_numpy().astype(str)

# ---------- 5. Split by image, stratified by species ----------
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
tr_idx, te_idx = next(sgkf.split(X, y, groups=df["image_id"]))

assert not set(df["image_id"].iloc[tr_idx]) & set(df["image_id"].iloc[te_idx]), "image leakage!"
print("Train:", len(tr_idx), "| Test:", len(te_idx))

np.savez("data/processed.npz",
         X_tr=X[tr_idx], y_tr=y[tr_idx], X_te=X[te_idx], y_te=y[te_idx])
print("Saved data/processed.npz")