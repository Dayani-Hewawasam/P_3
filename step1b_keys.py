import json

with open("data/keypoints.json") as f:
    data = json.load(f)

print("Top-level keys:", list(data.keys()))

for key in data:
    value = data[key]
    print("\n---", key, "---")
    print("Type:", type(value).__name__, "| Length:", len(value))

if "annotations" in data:
    print("\nFirst annotation:")
    print(data["annotations"][0])

if "categories" in data:
    print("\nCategories:")
    print(data["categories"])