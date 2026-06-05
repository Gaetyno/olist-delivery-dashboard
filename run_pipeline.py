from src.ingestion.ingest import run_ingestion
from src.cleaning.clean import run_cleaning
from src.transformations.build_gold import run_gold_build
from src.ml.train_model import main as run_ml_training


def main() -> None:
    print("=" * 80)
    print("LANCEMENT DU PIPELINE COMPLET")
    print("=" * 80)

    print("\nÉTAPE 1 — INGESTION BRONZE")
    run_ingestion()

    print("\nÉTAPE 2 — NETTOYAGE SILVER")
    run_cleaning()

    print("\nÉTAPE 3 — CONSTRUCTION GOLD")
    run_gold_build()

    print("\nÉTAPE 4 — ENTRAÎNEMENT ML")
    run_ml_training()

    print("=" * 80)
    print("PIPELINE COMPLET TERMINÉ AVEC SUCCÈS")
    print("=" * 80)


if __name__ == "__main__":
    main()