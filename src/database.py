"""
Connexion Google Sheets, chargement / mise à jour des données, helpers cache.
"""
import json
import logging
import time
from collections import Counter

import pandas as pd

logger = logging.getLogger(__name__)

STATUT_VALIDE = "Fait et validé"

CACHE_COLUMNS = [
    "_ratio",
    "_ttr",
    "_long_phrases",
    "_signature_json",
    "_coherence_score",
    "_trigrams_json",
]

# Erreurs API Google considérées comme temporaires (retry)
RETRYABLE_STATUS_CODES = (503, 429, 500, 502, 504)
MAX_RETRIES = 4
INITIAL_BACKOFF = 2.0


def load_data(conn, max_retries: int = MAX_RETRIES) -> pd.DataFrame:
    """Charge les données et assure la présence des colonnes de cache.

    En cas d'indisponibilité temporaire de l'API Google (503, 429, etc.),
    réessaie avec backoff exponentiel.
    """
    last_exception: BaseException | None = None
    backoff = INITIAL_BACKOFF

    for attempt in range(max_retries):
        try:
            data = conn.read(ttl="0")
            data = data.astype(str).replace(["nan", "None", "<NA>"], "")
            for col in CACHE_COLUMNS:
                if col not in data.columns:
                    data[col] = ""
            return data
        except Exception as ex:  # noqa: BLE001
            last_exception = ex
            status = getattr(ex, "response", None)
            status_code = getattr(status, "status_code", None) if status else None
            if status_code in RETRYABLE_STATUS_CODES and attempt < max_retries - 1:
                logger.warning(
                    "API Google indisponible (code %s), nouvel essai dans %.1fs (tentative %d/%d)",
                    status_code,
                    backoff,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(backoff)
                backoff *= 2
            else:
                raise

    if last_exception is not None:
        raise last_exception
    raise RuntimeError("load_data: échec après toutes les tentatives")


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
            "moy. mots/phrase": str(row.get("_long_phrases", "") or "—"),
            "TTR": str(row.get("_ttr", "") or "—"),
            "alertes": "—",
        })
    return rows


def dataset_cache_stats(df_valid: pd.DataFrame) -> dict | None:
    """Calcule les statistiques agrégées depuis le cache (pas de spaCy).

    Lit _ratio, _ttr, _long_phrases et _coherence_score pour chaque fiche
    dont le cache est rempli. Calcule également un score de santé global 0-100.

    Args:
        df_valid: Sous-ensemble du DataFrame filtré sur STATUT_VALIDE.

    Returns:
        Dict avec clés "n", "ratio", "ttr", "phrases", "coherence",
        "health_score" ou None si aucune fiche n'a de cache.
    """
    cols_map = {
        "ratio":     "_ratio",
        "ttr":       "_ttr",
        "phrases":   "_long_phrases",
        "coherence": "_coherence_score",
    }
    parsed: dict[str, list[float]] = {k: [] for k in cols_map}
    for _, row in df_valid.iterrows():
        vals: dict[str, float] = {}
        for key, col in cols_map.items():
            raw = row.get(col, "") or ""
            if not raw:
                break
            try:
                vals[key] = float(raw)
            except (ValueError, TypeError):
                break
        else:
            for key, val in vals.items():
                parsed[key].append(val)

    n = len(parsed["ratio"])
    if n == 0:
        return None

    def _stats(values: list[float]) -> dict:
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / max(1, len(values))
        std = variance ** 0.5
        return {"mean": round(mean, 3), "std": round(std, 3), "values": values}

    problematic = flag_problematic_rows(df_valid)
    pct_ok = (n - len(problematic)) / n * 100

    mean_coh = sum(parsed["coherence"]) / len(parsed["coherence"])
    mean_ttr = sum(parsed["ttr"]) / len(parsed["ttr"])
    ttr_score = min(100.0, mean_ttr / 0.72 * 100)
    health = int(round(0.4 * mean_coh + 0.3 * ttr_score + 0.3 * pct_ok))

    return {
        "n": n,
        "ratio":     _stats(parsed["ratio"]),
        "ttr":       _stats(parsed["ttr"]),
        "phrases":   _stats(parsed["phrases"]),
        "coherence": _stats(parsed["coherence"]),
        "health_score": max(0, min(100, health)),
    }


def flag_problematic_rows(df_valid: pd.DataFrame) -> list[dict]:
    """Identifie les fiches avec des problèmes de qualité à partir du cache.

    Critères :
        - Cohérence < 45 → "Cohérence critique"
        - Type Expansion + ratio < 1.5 → "Expansion faible"
        - TTR < 0.50 → "Vocabulaire répétitif"

    Args:
        df_valid: Sous-ensemble du DataFrame filtré sur STATUT_VALIDE.

    Returns:
        Liste de dicts {"id", "type", "forme", "ton", "alertes": list[str]}.
        Seules les fiches ayant au moins une alerte et un cache complet sont incluses.
    """
    result = []
    for _, row in df_valid.iterrows():
        try:
            ratio = float(row.get("_ratio", "") or "")
            ttr   = float(row.get("_ttr", "") or "")
            score = float(row.get("_coherence_score", "") or "")
        except (ValueError, TypeError):
            continue
        alertes: list[str] = []
        if score < 45:
            alertes.append("Cohérence critique")
        if str(row.get("type", "")) == "Expansion" and ratio < 1.5:
            alertes.append("Expansion faible")
        if ttr < 0.50:
            alertes.append("Vocabulaire répétitif")
        if alertes:
            result.append({
                "id":     row.get("id", ""),
                "type":   row.get("type", ""),
                "forme":  row.get("forme", ""),
                "ton":    row.get("ton", ""),
                "alertes": alertes,
            })
    return result


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
