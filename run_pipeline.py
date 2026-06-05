from src.ingestion.ingest import run_ingestion
from src.cleaning.clean import run_cleaning
from src.transformations.build_gold import run_gold_build


def main() -> None:
    print("=" * 80)
    print("LANCEMENT DU PIPELINE COMPLET")
    print("=" * 80)

    run_ingestion()
    print()

    run_cleaning()
    print()

    run_gold_build()

    print("=" * 80)
    print("PIPELINE TERMINÉ AVEC SUCCÈS")
    print("=" * 80)


if __name__ == "__main__":
    main()