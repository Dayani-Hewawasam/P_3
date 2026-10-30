import json
import numpy as np
import pandas as pd

with open("data/keypoints.json") as f:
    data = json.load(f)

id2name = {c["id"]: c["name"] for c in data["categories"]}

rows = []
for a in data["annotations"]:
    kp = np.array(a["keypoints"], dtype=np.float32)   # shape (20, 3)
    rows.append({
        "image_id": a["image_id"],
        "file": data["images"][str(a["image_id"])],
        "species": id2name[a["category_id"]],
        "bbox": a["bbox"],
        "kpts": kp[:, :2],
        "vis": kp[:, 2],
        "n_visible": int((kp[:, 2] > 0).sum()),
    })

df = pd.DataFrame(rows)

print("Shape:", df.shape)
print("\nAnimals per species:")
print(df["species"].value_counts())
print("\nAverage visible keypoints per species:")
print(df.groupby("species")["n_visible"].mean().round(1))
print("\nVisible keypoints overall:")
print(df["n_visible"].describe())
print("\nMost animals in one image:", df.groupby("image_id").size().max())

# Which bbox format is it? Check how many visible keypoints fall inside the box.
def fraction_inside(fmt):
    inside, total = 0, 0
    for _, r in df.iterrows():
        x0, y0, a, b = r["bbox"]
        x1, y1 = (a, b) if fmt == "xyxy" else (x0 + a, y0 + b)
        pts = r["kpts"][r["vis"] > 0]
        inside += ((pts[:, 0] >= x0) & (pts[:, 0] <= x1) &
                   (pts[:, 1] >= y0) & (pts[:, 1] <= y1)).sum()
        total += len(pts)
    return inside / total

print("\nKeypoints inside box if xyxy:", round(fraction_inside("xyxy"), 3))
print("Keypoints inside box if xywh:", round(fraction_inside("xywh"), 3))