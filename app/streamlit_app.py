from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = PROJECT_ROOT / "data" / "gold"


st.set_page_config(
    page_title="Olist Delivery Performance",
    page_icon="📦",
    layout="wide",
)


def get_latest_gold_batch() -> Path:
    gold_batches = [
        path for path in GOLD_DIR.iterdir()
        if path.is_dir() and path.name.startswith("gold_")
    ]

    if not gold_batches:
        st.error("Aucun dossier Gold trouvé. Lance d'abord `python run_pipeline.py`.")
        st.stop()

    return max(gold_batches, key=lambda path: path.name)


@st.cache_data
def load_data(gold_batch_dir: Path):
    orders = pd.read_csv(gold_batch_dir / "orders_enriched.csv")
    state_kpis = pd.read_csv(gold_batch_dir / "delivery_kpis_by_state.csv")
    seller_kpis = pd.read_csv(gold_batch_dir / "delivery_kpis_by_seller.csv")
    category_kpis = pd.read_csv(gold_batch_dir / "delivery_kpis_by_category.csv")
    ml_dataset = pd.read_csv(gold_batch_dir / "ml_dataset.csv")

    return orders, state_kpis, seller_kpis, category_kpis, ml_dataset


gold_batch_dir = get_latest_gold_batch()
orders, state_kpis, seller_kpis, category_kpis, ml_dataset = load_data(gold_batch_dir)


st.title("📦 Olist Delivery Performance Dashboard")

st.caption(f"Données Gold utilisées : `{gold_batch_dir.name}`")

st.markdown(
    """
    Ce dashboard analyse la performance de livraison de la marketplace Olist, 
    avec un focus sur les retards de livraison et leur impact potentiel sur la satisfaction client.
    """
)


# -----------------------------
# KPIs globaux
# -----------------------------

st.header("1. Vue d'ensemble")

delivered_orders = orders[orders["order_status"] == "delivered"].copy()

total_orders = len(orders)
delivered_count = len(delivered_orders)
late_rate = delivered_orders["late_delivery"].mean() * 100
avg_delivery_days = delivered_orders["delivery_days"].mean()
avg_review_score = delivered_orders["review_score"].mean()
negative_review_rate = delivered_orders["negative_review"].mean() * 100

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Commandes totales", f"{total_orders:,.0f}")
col2.metric("Commandes livrées", f"{delivered_count:,.0f}")
col3.metric("Taux de retard", f"{late_rate:.2f}%")
col4.metric("Délai moyen", f"{avg_delivery_days:.1f} jours")
col5.metric("Note moyenne", f"{avg_review_score:.2f}/5")

st.metric("Taux d'avis négatifs", f"{negative_review_rate:.2f}%")


# -----------------------------
# Évolution mensuelle
# -----------------------------

st.header("2. Évolution des commandes")

if "purchase_year_month" in orders.columns:
    monthly_orders = (
        orders
        .groupby("purchase_year_month")
        .agg(orders_count=("order_id", "count"))
        .reset_index()
        .sort_values("purchase_year_month")
    )

    st.line_chart(
        monthly_orders,
        x="purchase_year_month",
        y="orders_count"
    )


# -----------------------------
# Retards par État client
# -----------------------------

st.header("3. Retards par État client")

state_display = (
    state_kpis
    .sort_values("late_rate", ascending=False)
    .head(15)
)

st.bar_chart(
    state_display,
    x="customer_state",
    y="late_rate"
)

st.dataframe(
    state_display[
        [
            "customer_state",
            "orders_count",
            "late_rate",
            "avg_delivery_days",
            "avg_delay_days",
            "avg_review_score",
            "negative_review_rate",
        ]
    ],
    use_container_width=True
)


# -----------------------------
# Retards par catégorie
# -----------------------------

st.header("4. Retards par catégorie produit")

min_orders_category = st.slider(
    "Nombre minimum de commandes par catégorie",
    min_value=10,
    max_value=1000,
    value=100,
    step=10,
)

category_display = (
    category_kpis[category_kpis["orders_count"] >= min_orders_category]
    .sort_values("late_rate", ascending=False)
    .head(15)
)

st.bar_chart(
    category_display,
    x="main_product_category",
    y="late_rate"
)

st.dataframe(
    category_display[
        [
            "main_product_category",
            "orders_count",
            "late_rate",
            "avg_delivery_days",
            "avg_delay_days",
            "avg_review_score",
            "negative_review_rate",
        ]
    ],
    use_container_width=True
)


# -----------------------------
# Retards par vendeur
# -----------------------------

st.header("5. Retards par vendeur")

st.markdown(
    """
    Les noms commerciaux des vendeurs ne sont pas disponibles dans le dataset. 
    L'analyse utilise donc les `seller_id` anonymisés.
    """
)

min_orders_seller = st.slider(
    "Nombre minimum de commandes par vendeur",
    min_value=10,
    max_value=500,
    value=50,
    step=10,
)

seller_display = (
    seller_kpis[seller_kpis["orders_count"] >= min_orders_seller]
    .sort_values("late_rate", ascending=False)
    .head(15)
    .copy()
)

seller_display["seller_label"] = seller_display["main_seller_id"].astype(str).str[:8]

st.bar_chart(
    seller_display,
    x="seller_label",
    y="late_rate"
)

st.dataframe(
    seller_display[
        [
            "main_seller_id",
            "main_seller_state",
            "orders_count",
            "late_rate",
            "avg_delivery_days",
            "avg_delay_days",
            "avg_review_score",
            "negative_review_rate",
        ]
    ],
    use_container_width=True
)


# -----------------------------
# Dataset ML
# -----------------------------

st.header("6. Dataset ML")

st.markdown(
    """
    Le dataset ML est généré automatiquement depuis la couche Gold. 
    Il servira à prédire si une commande risque d'être livrée en retard.
    """
)

col1, col2 = st.columns(2)

col1.metric("Lignes dataset ML", f"{len(ml_dataset):,.0f}")
col2.metric("Variables ML", f"{ml_dataset.shape[1] - 2}")

st.dataframe(ml_dataset.head(100), use_container_width=True)


st.success("Dashboard chargé avec succès.")