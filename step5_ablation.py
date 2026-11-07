import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

d = np.load("data/processed.npz")
X_tr, y_tr, X_te, y_te = d["X_tr"], d["y_tr"], d["X_te"], d["y_te"]

variants = {
    "Coordinates + flags (60 features)": slice(0, 60),
    "Coordinates only (40 features)": slice(0, 40),
    "Flags only (20 features)": slice(40, 60),
}

print(f"{'Features':<36}{'Accuracy':>10}{'Macro F1':>10}")
for name, s in variants.items():
    m = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               random_state=42, n_jobs=-1)
    m.fit(X_tr[:, s], y_tr)
    p = m.predict(X_te[:, s])
    print(f"{name:<36}{accuracy_score(y_te, p):>10.3f}"
          f"{f1_score(y_te, p, average='macro'):>10.3f}")