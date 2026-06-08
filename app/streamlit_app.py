from pathlib import Path

import pandas as pd
import plotly.express as px
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
def load_orders(gold_batch_dir_str: str) -> pd.DataFrame:
    gold_batch_dir = Path(gold_batch_dir_str)
    orders = pd.read_csv(gold_batch_dir / "orders_enriched.csv")

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"],
        errors="coerce",
    )

    for col in ["late_delivery", "negative_review"]:
        if col in orders.columns:
            orders[col] = (
                orders[col]
                .astype(str)
                .str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
            )

    return orders


def format_percent(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f}%"


def format_number(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.0f}"


def format_float(value: float, suffix: str = "") -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f}{suffix}"


def build_dimension_analysis(df: pd.DataFrame, analysis_type: str) -> pd.DataFrame:
    delivered = df[df["order_status"] == "delivered"].copy()

    if analysis_type == "État client":
        group_cols = ["customer_state"]
        label_col = "customer_state"

    elif analysis_type == "Catégorie produit":
        group_cols = ["main_product_category"]
        label_col = "main_product_category"

    else:
        group_cols = ["main_seller_id", "main_seller_state"]
        label_col = "seller_label"

    analysis = (
        delivered
        .groupby(group_cols, dropna=False)
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

    analysis["late_rate"] = analysis["late_rate"] * 100
    analysis["negative_review_rate"] = analysis["negative_review_rate"] * 100

    if analysis_type == "Vendeur":
        analysis["seller_label"] = (
            analysis["main_seller_id"].astype(str).str[:8]
            + " / "
            + analysis["main_seller_state"].astype(str)
        )

    analysis["label"] = analysis[label_col].astype(str)

    analysis["priority_score"] = (
        analysis["orders_count"]
        * (analysis["late_rate"] / 100)
        * (analysis["negative_review_rate"] / 100)
    )

    return analysis


gold_batch_dir = get_latest_gold_batch()
orders = load_orders(str(gold_batch_dir))


# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------

st.sidebar.title("Filtres")

st.sidebar.caption(f"Données utilisées : `{gold_batch_dir.name}`")

min_date = orders["order_purchase_timestamp"].min().date()
max_date = orders["order_purchase_timestamp"].max().date()

date_range = st.sidebar.date_input(
    "Période d'achat",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if len(date_range) != 2:
    st.warning("Sélectionne une date de début et une date de fin.")
    st.stop()

start_date, end_date = date_range

states = sorted(orders["customer_state"].dropna().unique().tolist())
selected_states = st.sidebar.multiselect(
    "État client",
    options=states,
    default=[],
)

categories = sorted(orders["main_product_category"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Catégorie produit",
    options=categories,
    default=[],
)

analysis_type = st.sidebar.radio(
    "Analyse détaillée",
    ["État client", "Catégorie produit", "Vendeur"],
)

sort_options = {
    "Taux de retard": "late_rate",
    "Nombre de commandes": "orders_count",
    "Taux d'avis négatifs": "negative_review_rate",
    "Note moyenne": "avg_review_score",
    "Score de priorité": "priority_score",
}

sort_label = st.sidebar.selectbox(
    "Trier par",
    list(sort_options.keys()),
)

sort_col = sort_options[sort_label]

top_n = st.sidebar.slider(
    "Top N à afficher",
    min_value=5,
    max_value=30,
    value=15,
    step=5,
)

min_orders = st.sidebar.slider(
    "Minimum de commandes",
    min_value=1,
    max_value=500,
    value=50,
    step=10,
)


# -------------------------------------------------------------------
# FILTER DATA
# -------------------------------------------------------------------

filtered_orders = orders[
    (orders["order_purchase_timestamp"].dt.date >= start_date)
    & (orders["order_purchase_timestamp"].dt.date <= end_date)
].copy()

if selected_states:
    filtered_orders = filtered_orders[
        filtered_orders["customer_state"].isin(selected_states)
    ]

if selected_categories:
    filtered_orders = filtered_orders[
        filtered_orders["main_product_category"].isin(selected_categories)
    ]

delivered_orders = filtered_orders[
    filtered_orders["order_status"] == "delivered"
].copy()


# -------------------------------------------------------------------
# HEADER
# -------------------------------------------------------------------

st.title("📦 Olist Delivery Performance Dashboard")

st.markdown(
    """
    Dashboard interactif pour analyser les retards de livraison, leur impact sur la satisfaction client,
    et les segments opérationnels à surveiller.
    """
)

if filtered_orders.empty:
    st.error("Aucune donnée ne correspond aux filtres sélectionnés.")
    st.stop()


# -------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------

tab_exec, tab_impact, tab_analysis, tab_alerts, tab_data = st.tabs(
    [
        "Vue exécutive",
        "Impact satisfaction",
        "Analyse détaillée",
        "Alertes opérationnelles",
        "Données",
    ]
)


# -------------------------------------------------------------------
# TAB 1 — EXECUTIVE VIEW
# -------------------------------------------------------------------

with tab_exec:
    st.header("Vue exécutive")

    total_orders = len(filtered_orders)
    delivered_count = len(delivered_orders)
    late_rate = delivered_orders["late_delivery"].mean() * 100
    avg_delivery_days = delivered_orders["delivery_days"].mean()
    avg_review_score = delivered_orders["review_score"].mean()
    negative_review_rate = delivered_orders["negative_review"].mean() * 100

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Commandes", format_number(total_orders))
    col2.metric("Livrées", format_number(delivered_count))
    col3.metric("Taux de retard", format_percent(late_rate))
    col4.metric("Délai moyen", format_float(avg_delivery_days, " jours"))
    col5.metric("Note moyenne", format_float(avg_review_score, "/5"))

    st.metric("Taux d'avis négatifs", format_percent(negative_review_rate))

    st.subheader("Évolution mensuelle des commandes")

    monthly_orders = (
        filtered_orders
        .dropna(subset=["order_purchase_timestamp"])
        .assign(
            purchase_year_month=lambda df: df["order_purchase_timestamp"]
            .dt.to_period("M")
            .astype(str)
        )
        .groupby("purchase_year_month")
        .agg(orders_count=("order_id", "count"))
        .reset_index()
        .sort_values("purchase_year_month")
    )

    fig_monthly = px.line(
        monthly_orders,
        x="purchase_year_month",
        y="orders_count",
        markers=True,
        title="Nombre de commandes par mois",
    )

    st.plotly_chart(fig_monthly, use_container_width=True)


# -------------------------------------------------------------------
# TAB 2 — IMPACT SATISFACTION
# -------------------------------------------------------------------

with tab_impact:
    st.header("Impact du retard sur la satisfaction client")

    impact = (
        delivered_orders
        .dropna(subset=["late_delivery"])
        .groupby("late_delivery")
        .agg(
            orders_count=("order_id", "count"),
            avg_review_score=("review_score", "mean"),
            negative_review_rate=("negative_review", "mean"),
        )
        .reset_index()
    )

    impact["negative_review_rate"] = impact["negative_review_rate"] * 100
    impact["delivery_status"] = impact["late_delivery"].map(
        {
            False: "À l'heure ou en avance",
            True: "En retard",
        }
    )

    col1, col2 = st.columns(2)

    fig_review = px.bar(
        impact,
        x="delivery_status",
        y="avg_review_score",
        title="Note moyenne selon le statut de livraison",
        text_auto=".2f",
    )

    fig_negative = px.bar(
        impact,
        x="delivery_status",
        y="negative_review_rate",
        title="Taux d'avis négatifs selon le statut de livraison",
        text_auto=".2f",
    )

    col1.plotly_chart(fig_review, use_container_width=True)
    col2.plotly_chart(fig_negative, use_container_width=True)

    st.dataframe(
        impact[
            [
                "delivery_status",
                "orders_count",
                "avg_review_score",
                "negative_review_rate",
            ]
        ],
        use_container_width=True,
    )


# -------------------------------------------------------------------
# TAB 3 — DETAILED ANALYSIS
# -------------------------------------------------------------------

with tab_analysis:
    st.header(f"Analyse détaillée — {analysis_type}")

    analysis = build_dimension_analysis(filtered_orders, analysis_type)

    analysis_filtered = (
        analysis[analysis["orders_count"] >= min_orders]
        .sort_values(sort_col, ascending=False)
        .head(top_n)
        .copy()
    )

    if analysis_filtered.empty:
        st.warning("Aucun segment ne correspond aux filtres sélectionnés.")
    else:
        fig_analysis = px.bar(
            analysis_filtered.sort_values(sort_col, ascending=True),
            x=sort_col,
            y="label",
            orientation="h",
            title=f"Top {top_n} — {analysis_type} par {sort_label}",
            hover_data=[
                "orders_count",
                "late_rate",
                "avg_delivery_days",
                "avg_review_score",
                "negative_review_rate",
                "priority_score",
            ],
        )

        st.plotly_chart(fig_analysis, use_container_width=True)

        st.dataframe(
            analysis_filtered[
                [
                    "label",
                    "orders_count",
                    "late_rate",
                    "avg_delivery_days",
                    "avg_delay_days",
                    "avg_review_score",
                    "negative_review_rate",
                    "priority_score",
                ]
            ],
            use_container_width=True,
        )


# -------------------------------------------------------------------
# TAB 4 — OPERATIONAL ALERTS
# -------------------------------------------------------------------

with tab_alerts:
    st.header("Alertes opérationnelles")

    st.markdown(
        """
        Le score de priorité combine le volume de commandes, le taux de retard
        et le taux d'avis négatifs. Il permet d'éviter de prioriser uniquement
        des segments avec très peu de commandes.
        """
    )

    state_analysis = build_dimension_analysis(filtered_orders, "État client")
    category_analysis = build_dimension_analysis(filtered_orders, "Catégorie produit")
    seller_analysis = build_dimension_analysis(filtered_orders, "Vendeur")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("États à surveiller")
        st.dataframe(
            state_analysis[state_analysis["orders_count"] >= min_orders]
            .sort_values("priority_score", ascending=False)
            .head(5)[
                [
                    "label",
                    "orders_count",
                    "late_rate",
                    "negative_review_rate",
                    "priority_score",
                ]
            ],
            use_container_width=True,
        )

    with col2:
        st.subheader("Catégories à surveiller")
        st.dataframe(
            category_analysis[category_analysis["orders_count"] >= min_orders]
            .sort_values("priority_score", ascending=False)
            .head(5)[
                [
                    "label",
                    "orders_count",
                    "late_rate",
                    "negative_review_rate",
                    "priority_score",
                ]
            ],
            use_container_width=True,
        )

    with col3:
        st.subheader("Vendeurs à surveiller")
        st.dataframe(
            seller_analysis[seller_analysis["orders_count"] >= min_orders]
            .sort_values("priority_score", ascending=False)
            .head(5)[
                [
                    "label",
                    "orders_count",
                    "late_rate",
                    "negative_review_rate",
                    "priority_score",
                ]
            ],
            use_container_width=True,
        )


# -------------------------------------------------------------------
# TAB 5 — DATA
# -------------------------------------------------------------------

with tab_data:
    st.header("Données filtrées")

    st.write(f"Nombre de lignes après filtres : {len(filtered_orders):,}")

    columns_to_display = [
        "order_id",
        "order_status",
        "customer_state",
        "main_product_category",
        "main_seller_id",
        "late_delivery",
        "delivery_days",
        "delay_days",
        "review_score",
        "negative_review",
        "total_price",
        "total_freight",
        "payment_type",
    ]

    available_columns = [
        col for col in columns_to_display
        if col in filtered_orders.columns
    ]

    st.dataframe(
        filtered_orders[available_columns].head(500),
        use_container_width=True,
    )

st.success("Dashboard interactif chargé avec succès.")