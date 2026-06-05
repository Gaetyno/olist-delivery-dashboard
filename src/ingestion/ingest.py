from pathlib import Path
from datetime import datetime
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = PROJECT_ROOT / "data" / "source"
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze"


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


def check_source_files() -> None:
    """
    Vérifie que tous les fichiers CSV attendus sont présents dans data/source/.
    """
    missing_files = []

    for filename in FILES.values():
        file_path = SOURCE_DIR / filename

        if not file_path.exists():
            missing_files.append(filename)

    if missing_files:
        missing = "\n".join(missing_files)
        raise FileNotFoundError(
            f"Fichiers manquants dans {SOURCE_DIR} :\n{missing}"
        )


def create_bronze_batch_dir() -> Path:
    """
    Crée un dossier Bronze horodaté pour cette ingestion.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = BRONZE_DIR / f"ingestion_{timestamp}"

    batch_dir.mkdir(parents=True, exist_ok=True)

    return batch_dir


def copy_files_to_bronze(batch_dir: Path) -> None:
    """
    Copie les fichiers CSV bruts de data/source/ vers data/bronze/.
    Les fichiers ne sont pas modifiés.
    """
    for table_name, filename in FILES.items():
        source_path = SOURCE_DIR / filename
        target_path = batch_dir / filename

        shutil.copy2(source_path, target_path)

        print(f"{table_name:20s} -> {target_path}")


def run_ingestion() -> None:
    """
    Lance l'étape d'ingestion Bronze.
    """
    print("Démarrage de l'ingestion Bronze...")

    check_source_files()
    batch_dir = create_bronze_batch_dir()
    copy_files_to_bronze(batch_dir)

    print("Ingestion Bronze terminée.")
    print(f"Dossier créé : {batch_dir}")


if __name__ == "__main__":
    run_ingestion()