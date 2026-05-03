#!/usr/bin/env python3
"""
Importe un CSV (ex. export Google Sheet) dans la table PostgreSQL ``entries``.

Remplace tout le contenu actuel de ``entries`` (même sémantique qu'une sauvegarde app).

Usage:
    DATABASE_URL=postgresql://... uv run python scripts/import_csv_to_pg.py export.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Permet ``import src`` depuis la racine du dépôt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.database import create_db_engine, update_data

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Import CSV → PostgreSQL (table entries).")
    parser.add_argument("csv_path", help="Chemin vers le fichier CSV")
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", "").strip(),
        help="URL PostgreSQL (défaut : variable DATABASE_URL)",
    )
    args = parser.parse_args()
    if not args.database_url:
        raise SystemExit("DATABASE_URL ou --database-url est requis.")

    df = pd.read_csv(args.csv_path, dtype=str).fillna("")
    engine = create_db_engine(args.database_url)
    update_data(engine, df)
    logger.info("Import terminé (%d lignes).", len(df))


if __name__ == "__main__":
    main()
