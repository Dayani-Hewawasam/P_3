import numpy as np
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, ConfusionMatrixDisplay)

d = np.load("data/processed.npz")
X_tr, y_tr, X_te, y_te = d["X_tr"], d["y_tr"], d["X_te"], d["y_te"]

models = {
    "Majority class": DummyClassifier(strategy="most_frequent"),
    "Logistic regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            random_state=42, n_jobs=-1),
}

print(f"{'Model':<22}{'Accuracy':>10}{'Macro F1':>10}")
preds = {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    preds[name] = m.predict(X_te)
    print(f"{name:<22}{accuracy_score(y_te, preds[name]):>10.3f}"
          f"{f1_score(y_te, preds[name], average='macro'):>10.3f}")

print("\nRandom forest, per species:")
print(classification_report(y_te, preds["Random forest"]))

ConfusionMatrixDisplay.from_predictions(y_te, preds["Random forest"], xticks_rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png")