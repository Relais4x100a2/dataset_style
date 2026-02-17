"""
Connexion Google Sheets, chargement / mise à jour des données, helpers cache.
"""
import json
from collections import Counter

import pandas as pd

CACHE_COLUMNS = [
    "_ratio",
    "_richesse",
    "_ttr",
    "_long_phrases",
    "_signature_json",
    "_coherence_score",
    "_trigrams_json",
]


def load_data(conn) -> pd.DataFrame:
    """Charge les données et assure la présence des colonnes de cache."""
    data = conn.read(ttl="0")
    data = data.astype(str).replace(["nan", "None", "<NA>"], "")
    for col in CACHE_COLUMNS:
        if col not in data.columns:
            data[col] = ""
    return data


def update_data(conn, df: pd.DataFrame) -> None:
    """Met à jour le Google Sheet avec le DataFrame."""
    conn.update(data=df)


def avg_signature_from_cache(df_valid: pd.DataFrame) -> dict[str, float] | None:
    """
    Moyenne des signatures à partir des colonnes _signature_json.
    N'utilise pas spaCy — évite OOM sur Cloud.
    """
    sigs = []
    for _, row in df_valid.iterrows():
        raw = row.get("_signature_json", "") or ""
        if not raw:
            continue
        try:
            sigs.append(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            continue
    if not sigs:
        return None
    keys = list(sigs[0].keys())
    return {k: sum(s[k] for s in sigs) / len(sigs) for k in keys}


def audit_rows_from_cache(df_valid: pd.DataFrame) -> list[dict]:
    """Construit les lignes d'audit à partir des colonnes cache (pas de spaCy)."""
    rows = []
    for _, row in df_valid.iterrows():
        r_ratio = row.get("_ratio", "")
        if r_ratio == "" or r_ratio is None:
            continue
        try:
            ratio_val = float(r_ratio)
        except (ValueError, TypeError):
            continue
        rows.append({
            "id": row.get("id", ""),
            "type": row.get("type", ""),
            "ratio": round(ratio_val, 1),
            "richesse": str(row.get("_richesse", "") or "—"),
            "moy. mots/phrase": str(row.get("_long_phrases", "") or "—"),
            "TTR": str(row.get("_ttr", "") or "—"),
            "alertes": "—",
        })
    return rows


def avg_trigrams_from_cache(df_valid: pd.DataFrame) -> Counter | None:
    """Agrège les trigrammes POS à partir de _trigrams_json (pas de spaCy)."""
    total: Counter = Counter()
    for _, row in df_valid.iterrows():
        raw = row.get("_trigrams_json", "") or ""
        if not raw:
            continue
        try:
            total.update(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            continue
    return total if total else None
