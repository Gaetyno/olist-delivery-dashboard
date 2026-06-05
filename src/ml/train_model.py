from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GOLD_DIR = PROJECT_ROOT / "data" / "gold"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def get_latest_gold_batch() -> Path:
    gold_batches = [
        path for path in GOLD_DIR.iterdir()
        if path.is_dir() and path.name.startswith("gold_")
    ]

    if not gold_batches:
        raise FileNotFoundError(
            "Aucun dossier Gold trouvé dans data/gold/. "
            "Lance d'abord run_pipeline.py."
        )

    return max(gold_batches, key=lambda path: path.name)


def load_ml_dataset() -> pd.DataFrame:
    gold_batch_dir = get_latest_gold_batch()
    ml_path = gold_batch_dir / "ml_dataset.csv"

    if not ml_path.exists():
        raise FileNotFoundError(f"Fichier ML introuvable : {ml_path}")

    print(f"Dataset ML utilisé : {ml_path}")

    return pd.read_csv(ml_path)


def prepare_features_and_target(df: pd.DataFrame):
    target = "late_delivery"

    if target not in df.columns:
        raise ValueError(f"Colonne target absente : {target}")

    df = df.copy()

    # Sécurité : si late_delivery est lu comme texte, on le reconvertit en booléen.
    if df[target].dtype == "object":
        df[target] = df[target].map({
            "True": True,
            "False": False,
            True: True,
            False: False,
        })

    df = df.dropna(subset=[target])

    X = df.drop(columns=["order_id", target])
    y = df[target].astype(bool)

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    return X, y, numeric_features, categorical_features


def build_model(numeric_features, categorical_features) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    return model


def train_and_evaluate():
    df = load_ml_dataset()

    X, y, numeric_features, categorical_features = prepare_features_and_target(df)

    print("Taille du dataset :", df.shape)
    print("Variables numériques :", numeric_features)
    print("Variables catégorielles :", categorical_features)
    print()
    print("Répartition de la target :")
    print(y.value_counts(normalize=True).mul(100).round(2))
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_model(numeric_features, categorical_features)

    print("Entraînement du modèle Random Forest...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print()
    print("Classification report:")
    print(report)

    print("Confusion matrix:")
    print(matrix)

    return model, report, matrix


def save_model(model: Pipeline) -> Path:
    MODEL_DIR.mkdir(exist_ok=True)

    model_path = MODEL_DIR / "random_forest_late_delivery_baseline.joblib"

    joblib.dump(model, model_path)

    print(f"Modèle sauvegardé : {model_path}")

    return model_path


def save_metrics(report: str, matrix) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)

    metrics_path = REPORTS_DIR / "ml_baseline_metrics.txt"

    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write("Classification report\n")
        file.write("=====================\n\n")
        file.write(report)
        file.write("\n\nConfusion matrix\n")
        file.write("================\n\n")
        file.write(str(matrix))

    print(f"Métriques sauvegardées : {metrics_path}")

    return metrics_path


def main() -> None:
    print("=" * 80)
    print("ENTRAÎNEMENT DU MODÈLE ML BASELINE")
    print("=" * 80)

    model, report, matrix = train_and_evaluate()

    save_model(model)
    save_metrics(report, matrix)

    print("=" * 80)
    print("ENTRAÎNEMENT TERMINÉ")
    print("=" * 80)


if __name__ == "__main__":
    main()