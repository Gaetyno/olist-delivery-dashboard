from pathlib import Path
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BRONZE_DIR = PROJECT_ROOT / "data" / "bronze"
SILVER_DIR = PROJECT_ROOT / "data" / "silver"


FILES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": [
        "shipping_limit_date",
    ],
    "reviews": [
        "review_creation_date",
        "review_answer_timestamp",
    ],
}


def get_latest_bronze_batch() -> Path:
    """
    Récupère le dernier dossier d'ingestion Bronze.
    """
    bronze_batches = [
        path for path in BRONZE_DIR.iterdir()
        if path.is_dir() and path.name.startswith("ingestion_")
    ]

    if not bronze_batches:
        raise FileNotFoundError(
            "Aucun dossier d'ingestion trouvé dans data/bronze/. "
            "Lance d'abord src/ingestion/ingest.py."
        )

    latest_batch = max(bronze_batches, key=lambda path: path.name)

    return latest_batch


def create_silver_batch_dir() -> Path:
    """
    Crée un dossier Silver horodaté pour cette étape de nettoyage.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = SILVER_DIR / f"cleaning_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    return batch_dir


def load_csv_files(bronze_batch_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Charge les fichiers CSV depuis le dernier dossier Bronze.
    """
    data = {}

    for table_name, filename in FILES.items():
        file_path = bronze_batch_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Fichier manquant : {file_path}")

        data[table_name] = pd.read_csv(file_path)

    return data


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise les noms de colonnes.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    return df


def remove_full_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les lignes entièrement dupliquées.
    """
    return df.drop_duplicates().copy()


def convert_date_columns(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Convertit les colonnes de dates connues en datetime.
    """
    df = df.copy()

    for col in DATE_COLUMNS.get(table_name, []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage spécifique de la table orders.
    """
    df = df.copy()

    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.days

    df["delay_days"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.days

    delivered_mask = (
        (df["order_status"] == "delivered")
        & df["order_delivered_customer_date"].notna()
        & df["order_estimated_delivery_date"].notna()
    )

    df["late_delivery"] = pd.NA
    df.loc[delivered_mask, "late_delivery"] = (
        df.loc[delivered_mask, "order_delivered_customer_date"]
        > df.loc[delivered_mask, "order_estimated_delivery_date"]
    )

    df["late_delivery"] = df["late_delivery"].astype("boolean")

    return df


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage spécifique de la table reviews.
    """
    df = df.copy()

    if "review_score" in df.columns:
        df["negative_review"] = df["review_score"] <= 2

    return df


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage spécifique de la table products.
    """
    df = df.copy()

    dimension_cols = [
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]

    if all(col in df.columns for col in dimension_cols):
        df["product_volume_cm3"] = (
            df["product_length_cm"]
            * df["product_height_cm"]
            * df["product_width_cm"]
        )

    return df


def clean_table(table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique le nettoyage standard puis les règles spécifiques par table.
    """
    df = standardize_columns(df)
    df = remove_full_duplicates(df)
    df = convert_date_columns(df, table_name)

    if table_name == "orders":
        df = clean_orders(df)

    if table_name == "reviews":
        df = clean_reviews(df)

    if table_name == "products":
        df = clean_products(df)

    return df


def save_silver_files(data: dict[str, pd.DataFrame], silver_batch_dir: Path) -> None:
    """
    Sauvegarde les tables nettoyées dans data/silver/.
    """
    for table_name, df in data.items():
        output_path = silver_batch_dir / f"{table_name}.csv"
        df.to_csv(output_path, index=False)

        print(f"{table_name:20s} -> {output_path}")


def run_cleaning() -> None:
    """
    Lance l'étape Silver.
    """
    print("Démarrage du nettoyage Silver...")

    bronze_batch_dir = get_latest_bronze_batch()
    print(f"Dossier Bronze utilisé : {bronze_batch_dir}")

    raw_data = load_csv_files(bronze_batch_dir)

    cleaned_data = {}

    for table_name, df in raw_data.items():
        cleaned_data[table_name] = clean_table(table_name, df)

    silver_batch_dir = create_silver_batch_dir()
    save_silver_files(cleaned_data, silver_batch_dir)

    print("Nettoyage Silver terminé.")
    print(f"Dossier créé : {silver_batch_dir}")


if __name__ == "__main__":
    run_cleaning()