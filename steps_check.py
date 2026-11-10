import json
import numpy as np
import pandas as pd

with open("data/keypoints.json") as f:
    data = json.load(f)

rows = []
for a in data["annotations"]:
    kp = np.array(a["keypoints"], dtype=np.float32)
    x0, y0, x1, y1 = a["bbox"]
    rows.append({"image_id": a["image_id"], "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                 "kpts": kp[:, :2], "vis": kp[:, 2]})
df = pd.DataFrame(rows)
df["n_visible"] = df["vis"].apply(lambda v: int((v > 0).sum()))

print("CHECK 1: how many dots do animals have?")
print("  animals with 0 dots:", (df["n_visible"] == 0).sum())
for t in [1, 3, 5, 8, 10]:
    kept = (df["n_visible"] >= t).sum()
    print(f"  keep animals with >= {t:>2} dots -> {kept} left ({kept / len(df):.0%})")

print("\nCHECK 2: do animals differ in size and position?")
df["w"] = df["x1"] - df["x0"]
df["h"] = df["y1"] - df["y0"]
print(df[["w", "h", "x0", "y0"]].describe().round(0))
print("  biggest box / smallest box (width):", round(df["w"].max() / max(df["w"].min(), 1), 1))

print("\nCHECK 3: what do missing dots look like?")
all_vis = np.concatenate(df["vis"].values)
all_xy = np.concatenate(df["kpts"].values)
missing = all_vis == 0
print(f"  share of all dots that are missing: {missing.mean():.0%}")
print("  missing dots stored as (0, 0)?", bool((all_xy[missing] == 0).all()))

# after centering, how many REAL dots land almost exactly at the center (0, 0)?
near_center = total = 0
for _, r in df[df["n_visible"] >= 5].iterrows():
    w, h = max(r["x1"] - r["x0"], 1e-6), max(r["y1"] - r["y0"], 1e-6)
    s = max(w, h)
    cx, cy = (r["x0"] + r["x1"]) / 2, (r["y0"] + r["y1"]) / 2
    p = r["kpts"][r["vis"] > 0]
    nx, ny = (p[:, 0] - cx) / s, (p[:, 1] - cy) / s
    near_center += ((abs(nx) < 0.02) & (abs(ny) < 0.02)).sum()
    total += len(p)
print(f"  real dots within 0.02 of the center after normalizing: {near_center} of {total}")

print("\nCHECK 4: do animals share photos?")
per_img = df.groupby("image_id").size()
print("  photos with more than one animal:", (per_img > 1).sum(), "of", len(per_img))
print("  most animals in one photo:", per_img.max())