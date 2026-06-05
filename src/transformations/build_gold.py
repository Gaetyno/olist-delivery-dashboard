from pathlib import Path
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SILVER_DIR = PROJECT_ROOT / "data" / "silver"
GOLD_DIR = PROJECT_ROOT / "data" / "gold"


def get_latest_silver_batch() -> Path:
    silver_batches = [
        path for path in SILVER_DIR.iterdir()
        if path.is_dir() and path.name.startswith("cleaning_")
    ]

    if not silver_batches:
        raise FileNotFoundError(
            "Aucun dossier Silver trouvé dans data/silver/. "
            "Lance d'abord src/cleaning/clean.py."
        )

    return max(silver_batches, key=lambda path: path.name)


def create_gold_batch_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = GOLD_DIR / f"gold_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir


def load_silver_files(silver_batch_dir: Path) -> dict:
    files = {
        "customers": "customers.csv",
        "orders": "orders.csv",
        "order_items": "order_items.csv",
        "payments": "payments.csv",
        "reviews": "reviews.csv",
        "products": "products.csv",
        "sellers": "sellers.csv",
        "category_translation": "category_translation.csv",
    }

    data = {}

    for table_name, filename in files.items():
        path = silver_batch_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"Fichier manquant : {path}")

        data[table_name] = pd.read_csv(path)

    return data


def convert_dates(orders: pd.DataFrame) -> pd.DataFrame:
    orders = orders.copy()

    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for col in date_cols:
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col], errors="coerce")

    return orders


def build_orders_enriched(data: dict) -> pd.DataFrame:
    orders = convert_dates(data["orders"])
    customers = data["customers"]
    order_items = data["order_items"]
    payments = data["payments"]
    reviews = data["reviews"]
    products = data["products"]
    sellers = data["sellers"]
    category_translation = data["category_translation"]

    # Items agrégés au niveau commande
    items_by_order = (
        order_items
        .groupby("order_id")
        .agg(
            item_count=("order_item_id", "count"),
            product_count=("product_id", "nunique"),
            seller_count=("seller_id", "nunique"),
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            avg_price=("price", "mean"),
        )
        .reset_index()
    )

    # Vendeur principal de la commande : celui associé au plus gros prix
    main_seller = (
        order_items
        .sort_values("price", ascending=False)
        .drop_duplicates("order_id")
        [["order_id", "seller_id"]]
        .merge(
            sellers[["seller_id", "seller_city", "seller_state"]],
            on="seller_id",
            how="left"
        )
        .rename(columns={
            "seller_id": "main_seller_id",
            "seller_city": "main_seller_city",
            "seller_state": "main_seller_state",
        })
    )

    # Catégorie principale de la commande
    products_translated = products.merge(
        category_translation,
        on="product_category_name",
        how="left"
    )

    items_with_category = order_items.merge(
        products_translated[["product_id", "product_category_name_english"]],
        on="product_id",
        how="left"
    )

    main_category = (
        items_with_category
        .dropna(subset=["product_category_name_english"])
        .drop_duplicates("order_id")
        [["order_id", "product_category_name_english"]]
        .rename(columns={"product_category_name_english": "main_product_category"})
    )

    # Paiements agrégés au niveau commande
    payments_by_order = (
        payments
        .groupby("order_id")
        .agg(
            payment_value=("payment_value", "sum"),
            payment_installments=("payment_installments", "max"),
            payment_types_count=("payment_type", "nunique"),
        )
        .reset_index()
    )

    main_payment_type = (
        payments
        .sort_values("payment_value", ascending=False)
        .drop_duplicates("order_id")
        [["order_id", "payment_type"]]
    )

    # Avis agrégés au niveau commande
    reviews_by_order = (
        reviews
        .groupby("order_id")
        .agg(
            review_score=("review_score", "mean"),
            negative_review=("negative_review", "max"),
        )
        .reset_index()
    )

    orders_enriched = (
        orders
        .merge(customers, on="customer_id", how="left")
        .merge(items_by_order, on="order_id", how="left")
        .merge(main_seller, on="order_id", how="left")
        .merge(main_category, on="order_id", how="left")
        .merge(payments_by_order, on="order_id", how="left")
        .merge(main_payment_type, on="order_id", how="left")
        .merge(reviews_by_order, on="order_id", how="left")
    )

    orders_enriched["purchase_month"] = orders_enriched["order_purchase_timestamp"].dt.month
    orders_enriched["purchase_dayofweek"] = orders_enriched["order_purchase_timestamp"].dt.dayofweek
    orders_enriched["purchase_year_month"] = orders_enriched["order_purchase_timestamp"].dt.to_period("M").astype(str)

    orders_enriched["delivery_days"] = (
        orders_enriched["order_delivered_customer_date"]
        - orders_enriched["order_purchase_timestamp"]
    ).dt.days

    orders_enriched["main_product_category"] = orders_enriched["main_product_category"].fillna("unknown")

    return orders_enriched


def build_delivery_kpis_by_state(orders_enriched: pd.DataFrame) -> pd.DataFrame:
    delivered = orders_enriched[orders_enriched["order_status"] == "delivered"].copy()

    kpis = (
        delivered
        .groupby("customer_state")
        .agg(
            orders_count=("order_id", "count"),
            late_rate=("late_delivery", "mean"),
            avg_delivery_days=("delivery_days", "mean"),
            avg_delay_days=("delay_days", "mean"),
            avg_review_score=("review_score", "mean"),
            negative_review_rate=("negative_review", "mean"),
        )
        .reset_index()
    )

    kpis["late_rate"] = kpis["late_rate"] * 100
    kpis["negative_review_rate"] = kpis["negative_review_rate"] * 100

    return kpis.sort_values("orders_count", ascending=False)


def build_delivery_kpis_by_seller(orders_enriched: pd.DataFrame) -> pd.DataFrame:
    delivered = orders_enriched[orders_enriched["order_status"] == "delivered"].copy()

    kpis = (
        delivered
        .groupby(["main_seller_id", "main_seller_state"])
        .agg(
            orders_count=("order_id", "count"),
            late_rate=("late_delivery", "mean"),
            avg_delivery_days=("delivery_days", "mean"),
            avg_delay_days=("delay_days", "mean"),
            avg_review_score=("review_score", "mean"),
            negative_review_rate=("negative_review", "mean"),
        )
        .reset_index()
    )

    kpis["late_rate"] = kpis["late_rate"] * 100
    kpis["negative_review_rate"] = kpis["negative_review_rate"] * 100

    return kpis.sort_values("orders_count", ascending=False)


def build_delivery_kpis_by_category(orders_enriched: pd.DataFrame) -> pd.DataFrame:
    delivered = orders_enriched[orders_enriched["order_status"] == "delivered"].copy()

    kpis = (
        delivered
        .groupby("main_product_category")
        .agg(
            orders_count=("order_id", "count"),
            late_rate=("late_delivery", "mean"),
            avg_delivery_days=("delivery_days", "mean"),
            avg_delay_days=("delay_days", "mean"),
            avg_review_score=("review_score", "mean"),
            negative_review_rate=("negative_review", "mean"),
        )
        .reset_index()
    )

    kpis["late_rate"] = kpis["late_rate"] * 100
    kpis["negative_review_rate"] = kpis["negative_review_rate"] * 100

    return kpis.sort_values("orders_count", ascending=False)


def build_ml_dataset(orders_enriched: pd.DataFrame) -> pd.DataFrame:
    ml_dataset = orders_enriched[
        (orders_enriched["order_status"] == "delivered")
        & (orders_enriched["late_delivery"].notna())
    ].copy()

    selected_columns = [
        "order_id",
        "late_delivery",
        "customer_state",
        "main_seller_state",
        "main_product_category",
        "item_count",
        "product_count",
        "seller_count",
        "total_price",
        "total_freight",
        "payment_value",
        "payment_installments",
        "payment_type",
        "estimated_delivery_days",
        "purchase_month",
        "purchase_dayofweek",
    ]

    ml_dataset = ml_dataset[selected_columns].copy()

    return ml_dataset


def save_gold_table(df: pd.DataFrame, gold_batch_dir: Path, filename: str) -> None:
    output_path = gold_batch_dir / filename
    df.to_csv(output_path, index=False)
    print(f"{filename:35s} -> {output_path}")


def run_gold_build() -> None:
    print("Démarrage de la construction Gold...")

    silver_batch_dir = get_latest_silver_batch()
    print(f"Dossier Silver utilisé : {silver_batch_dir}")

    data = load_silver_files(silver_batch_dir)

    gold_batch_dir = create_gold_batch_dir()

    orders_enriched = build_orders_enriched(data)
    delivery_kpis_by_state = build_delivery_kpis_by_state(orders_enriched)
    delivery_kpis_by_seller = build_delivery_kpis_by_seller(orders_enriched)
    delivery_kpis_by_category = build_delivery_kpis_by_category(orders_enriched)
    ml_dataset = build_ml_dataset(orders_enriched)

    save_gold_table(orders_enriched, gold_batch_dir, "orders_enriched.csv")
    save_gold_table(delivery_kpis_by_state, gold_batch_dir, "delivery_kpis_by_state.csv")
    save_gold_table(delivery_kpis_by_seller, gold_batch_dir, "delivery_kpis_by_seller.csv")
    save_gold_table(delivery_kpis_by_category, gold_batch_dir, "delivery_kpis_by_category.csv")
    save_gold_table(ml_dataset, gold_batch_dir, "ml_dataset.csv")

    print("Construction Gold terminée.")
    print(f"Dossier créé : {gold_batch_dir}")


if __name__ == "__main__":
    run_gold_build()