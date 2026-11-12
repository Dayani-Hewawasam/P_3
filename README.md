python -m venv venv
venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib

python step3_prepare.py     # cleans, normalizes, splits, saves data/processed.npz
python step4_train.py       # trains the three models, prints scores, saves confusion_matrix.png
python step5_ablation.py    # compares feature sets