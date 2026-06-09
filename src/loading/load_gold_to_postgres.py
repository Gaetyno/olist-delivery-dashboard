import os
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GOLD_DIR = PROJECT_ROOT / "data" / "gold"

TABLES = {
    "orders_enriched": "orders_enriched.csv",
    "delivery_kpis_by_state": "delivery_kpis_by_state.csv",
    "delivery_kpis_by_seller": "delivery_kpis_by_seller.csv",
    "delivery_kpis_by_category": "delivery_kpis_by_category.csv",
    "ml_dataset": "ml_dataset.csv",
}


def get_latest_gold_batch() -> Path:
    """
    Récupère le dernier dossier Gold généré par le pipeline.
    """
    gold_batches = [
        path for path in GOLD_DIR.iterdir()
        if path.is_dir() and path.name.startswith("gold_")
    ]

    if not gold_batches:
        raise FileNotFoundError(
            "Aucun dossier Gold trouvé dans data/gold/. "
            "Lance d'abord python run_pipeline.py."
        )

    return max(gold_batches, key=lambda path: path.name)


def get_database_url() -> str:
    """
    Construit l'URL de connexion PostgreSQL depuis le fichier .env.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    db = os.getenv("POSTGRES_DB")

    missing_vars = [
        name for name, value in {
            "POSTGRES_USER": user,
            "POSTGRES_PASSWORD": password,
            "POSTGRES_HOST": host,
            "POSTGRES_PORT": port,
            "POSTGRES_DB": db,
        }.items()
        if not value
    ]

    if missing_vars:
        raise ValueError(
            "Variables manquantes dans .env : "
            + ", ".join(missing_vars)
        )

    password_encoded = quote_plus(password)

    return f"postgresql+psycopg2://{user}:{password_encoded}@{host}:{port}/{db}"


def create_postgres_engine():
    """
    Crée la connexion SQLAlchemy vers PostgreSQL.
    """
    database_url = get_database_url()
    engine = create_engine(database_url)

    return engine


def test_connection(engine) -> None:
    """
    Vérifie que la connexion PostgreSQL fonctionne.
    """
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        result.scalar()

    print("Connexion PostgreSQL réussie.")


def load_table_to_postgres(
    engine,
    csv_path: Path,
    table_name: str,
) -> None:
    """
    Charge un fichier CSV Gold dans une table PostgreSQL.
    Si la table existe déjà, elle est remplacée.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=10_000,
    )

    print(f"{table_name:35s} -> {len(df):,} lignes chargées")


def load_gold_to_postgres() -> None:
    """
    Charge toutes les tables Gold dans PostgreSQL.
    """
    print("=" * 80)
    print("CHARGEMENT GOLD VERS POSTGRESQL")
    print("=" * 80)

    gold_batch_dir = get_latest_gold_batch()
    print(f"Dossier Gold utilisé : {gold_batch_dir}")

    engine = create_postgres_engine()
    test_connection(engine)

    for table_name, filename in TABLES.items():
        csv_path = gold_batch_dir / filename
        load_table_to_postgres(engine, csv_path, table_name)

    print("=" * 80)
    print("CHARGEMENT POSTGRESQL TERMINÉ")
    print("=" * 80)


if __name__ == "__main__":
    load_gold_to_postgres()