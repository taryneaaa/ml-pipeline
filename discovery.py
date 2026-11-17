"""Model discovery: compare a low-code PyCaret workflow with a manual scikit-learn one.

Run with:  uv run python discovery.py
"""

import shutil
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # write images to disk; never open a window
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path("data/telco_churn.csv")
IMAGES = Path("images")
IMAGES.mkdir(exist_ok=True)
TARGET = "Churn"

df = pd.read_csv(DATA_PATH)
print(f"Loaded {DATA_PATH}: {df.shape[0]} rows, {df.shape[1]} columns")


def save_table_image(table: pd.DataFrame, path: Path, title: str) -> None:
    """Render a DataFrame to a PNG so the comparison table is saved as an image."""
    fig, ax = plt.subplots(figsize=(14, 0.3 * len(table) + 0.6))
    ax.axis("off")
    ax.set_title(title, fontsize=12, loc="left")
    widths = [0.28] + [0.09] * (table.shape[1] - 1)
    tbl = ax.table(
        cellText=table.values,
        colLabels=table.columns,
        rowLabels=table.index,
        colWidths=widths,
        loc="upper center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. PyCaret workflow
# ---------------------------------------------------------------------------
from pycaret.classification import compare_models, plot_model, pull, setup

pycaret_start = time.perf_counter()

setup(data=df, target=TARGET, session_id=42, verbose=False)
top3 = compare_models(n_select=3)
comparison = pull()  # the table compare_models printed
best = top3[0]
plot_model(best, plot="confusion_matrix", save=True)

pycaret_seconds = time.perf_counter() - pycaret_start

print("\nPyCaret comparison (10-fold CV on the training split):")
print(comparison.to_string())
print(f"\nTop 3 models: {[type(m).__name__ for m in top3]}")
print(f"PyCaret section took {pycaret_seconds:.1f} s")

save_table_image(
    comparison.drop(columns=["TT (Sec)"], errors="ignore").round(4),
    IMAGES / "pycaret_compare_models.png",
    "PyCaret compare_models() - Telco churn, session_id=42",
)
shutil.move("Confusion Matrix.png", IMAGES / "pycaret_confusion_matrix.png")

# ---------------------------------------------------------------------------
# 2. scikit-learn workflow: rebuild PyCaret's best model by hand
# ---------------------------------------------------------------------------
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sklearn_start = time.perf_counter()

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric_cols,
        ),
        (
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_cols,
        ),
    ]
)

# PyCaret's best model was Logistic Regression; same settings PyCaret uses.
model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

sklearn_seconds = time.perf_counter() - sklearn_start

print("\nscikit-learn classification report (30% held-out test set):")
print(classification_report(y_test, y_pred, digits=4))
print(f"scikit-learn section took {sklearn_seconds:.3f} s")

joblib.dump(model, "model.joblib")
print("Saved fitted pipeline to model.joblib")
