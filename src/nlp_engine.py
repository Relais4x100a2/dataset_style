"""
Moteur linguistique : insights linguistiques, stylométrie, cohérence.
Sans dépendance Streamlit — testable indépendamment.
"""
import json
import logging
from collections import Counter
from typing import Callable

import pandas as pd
import requests

logger = logging.getLogger(__name__)

LANGUAGETOOL_API_URL = "https://api.languagetool.org/v2/check"
LANGUAGETOOL_TIMEOUT = 15

VERBES_FAIBLES = {"être", "avoir", "faire", "aller", "dire"}

_POS_FR: dict[str, str] = {
    "ADJ": "Adj",
    "ADP": "Prép",
    "ADV": "Adv",
    "AUX": "Aux",
    "CCONJ": "Conj",
    "DET": "Dét",
    "INTJ": "Intj",
    "NOUN": "Nom",
    "NUM": "Num",
    "PART": "Part",
    "PRON": "Pron",
    "PROPN": "NomPr",
    "PUNCT": "Ponct",
    "SCONJ": "SubConj",
    "SYM": "Sym",
    "VERB": "Verbe",
    "X": "Autre",
}


def corriger_texte_fr(text: str) -> str:
    """
    Corrige l'orthographe et la grammaire du texte en français via l'API
    LanguageTool (pas de réécriture, uniquement corrections ciblées).

    Args:
        text: Texte à corriger.

    Returns:
        Le texte avec les corrections appliquées. Retourne une chaîne vide
        si text est vide.

    Raises:
        requests.RequestException: En cas de timeout ou d'erreur réseau.
        ValueError: Si la réponse de l'API est invalide.
    """
    if not text or not text.strip():
        return ""

    try:
        resp = requests.post(
            LANGUAGETOOL_API_URL,
            data={"text": text, "language": "fr"},
            timeout=LANGUAGETOOL_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.Timeout:
        logger.warning("LanguageTool API timeout")
        raise
    except requests.RequestException as e:
        logger.warning("LanguageTool API error: %s", e)
        raise

    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        logger.warning("LanguageTool API invalid JSON: %s", e)
        raise ValueError("Réponse API invalide") from e

    matches = data.get("matches", [])
    if not matches:
        return text

    # Appliquer les corrections de la fin vers le début pour ne pas décaler les offsets
    result = text
    for match in sorted(matches, key=lambda m: m["offset"], reverse=True):
        offset = match["offset"]
        length = match["length"]
        replacements = match.get("replacements", [])
        if not replacements:
            continue
        replacement = replacements[0].get("value")
        if replacement is None:
            continue
        result = result[:offset] + replacement + result[offset + length :]

    return result


def get_linguistic_insights(
    text_in: str, text_out: str, nlp, seuil_repetition: int = 3
) -> dict | None:
    """
    Analyse linguistique input/output : ratio d'expansion, richesse lexicale,
    TTR, mots répétés, longueur moyenne des phrases.
    Retourne None si nlp est None ou textes vides.
    """
    if nlp is None or not (text_in and text_out):
        return None
    doc_in = nlp(text_in)
    doc_out = nlp(text_out)
    tokens_in = [t for t in doc_in if not t.is_punct]
    tokens_out = [t for t in doc_out if not t.is_punct]
    len_in = len(tokens_in)
    len_out = len(tokens_out)
    ratio = len_out / max(1, len_in)

    lemmes_out = {t.lemma_.lower() for t in doc_out if not t.is_punct}
    ttr = len(lemmes_out) / max(1, len_out)
    comptage = Counter(
        t.lemma_.lower() for t in doc_out if not t.is_punct and not t.is_stop
    )
    mots_repetes = [lem for lem, n in comptage.items() if n >= seuil_repetition]

    sents = list(doc_out.sents)
    long_phrases = [len([t for t in s if not t.is_punct]) for s in sents]
    long_moy_phrases = sum(long_phrases) / max(1, len(long_phrases))

    stop_in = sum(1 for t in doc_in if not t.is_punct and t.is_stop)
    stop_out = sum(1 for t in doc_out if not t.is_punct and t.is_stop)
    stop_ratio_in = stop_in / max(1, len_in)
    stop_ratio_out = stop_out / max(1, len_out)

    return {
        "ratio": ratio,
        "mots_in": len_in,
        "mots_out": len_out,
        "ttr": ttr,
        "mots_repetes": mots_repetes,
        "long_moy_phrases": long_moy_phrases,
        "stop_ratio_in": stop_ratio_in,
        "stop_ratio_out": stop_ratio_out,
    }


def get_baguette_touch(text_out: str, nlp) -> dict | None:
    """
    Indicateurs « Baguette-Touch » : ponctuation expressive, verbes faibles.
    Retourne None si nlp absent ou texte vide.
    """
    if nlp is None or not text_out:
        return None
    doc = nlp(text_out)
    text = text_out
    punct_exp = {
        "tiret_cadratin": text.count("—"),
        "points_suspension": text.count("..."),
        "deux_points": text.count(":"),
    }
    weak_verbs: list[tuple[str, int]] = []
    verb_counts = Counter(
        t.lemma_.lower() for t in doc if t.pos_ in ("VERB", "AUX")
    )
    for v in VERBES_FAIBLES:
        c = verb_counts.get(v, 0)
        if c > 0:
            weak_verbs.append((v, c))
    return {"punct_exp": punct_exp, "weak_verbs": weak_verbs}


def syntax_contrast_score(text_in: str, text_out: str, nlp) -> float:
    """
    Distance syntaxique Input vs Output (0–1). Élevé = output bien transformé.
    Bas = structures trop proches, le modèle n'apprendra pas grand-chose.
    """
    if nlp is None or not (text_in and text_out):
        return 0.0
    sig_in = get_stylometric_signature(text_in, nlp)
    sig_out = get_stylometric_signature(text_out, nlp)
    if not sig_in or not sig_out:
        return 0.0
    keys = list(sig_in.keys())
    total = 0.0
    for k in keys:
        total += abs((sig_in.get(k, 0) - sig_out.get(k, 0)))
    return min(1.0, total / max(1, len(keys)) * 2)


# Table de données des paliers : indicateur → [(seuil_max, niveau, interprétation), ...]
# Le dernier tuple (seuil_max=inf) est le palier par défaut (valeur >= dernier seuil).
_PALIERS: dict[str, list[tuple[float, str, str]]] = {
    "ratio": [
        (1.3, "Minimal",    "Tu restes proche du brouillon."),
        (2.0, "Progressif", "Tu développes."),
        (2.5, "Solide",     "Tu as bien développé l'idée."),
        (float("inf"), "Amplifié", "Tu déploies beaucoup."),
    ],
    "ttr": [
        (0.50, "Bas",           "Vocabulaire répétitif."),
        (0.65, "Intermédiaire", "Vocabulaire correct."),
        (0.80, "Élevé",         "Vocabulaire soutenu."),
        (float("inf"), "Très élevé", "Vocabulaire très riche."),
    ],
    "moy_phrases": [
        (10,  "Court",      "Rythme vif, phrases courtes."),
        (18,  "Équilibré",  "Rythme équilibré."),
        (25,  "Ample",      "Rythme ample."),
        (float("inf"), "Très ample", "Phrases très longues."),
    ],
}


def palier_details(indicateur: str, value: float) -> tuple[str, str]:
    """Retourne (niveau, interprétation) pour un indicateur et une valeur."""
    for seuil, niveau, interpretation in _PALIERS.get(indicateur, []):
        if value < seuil:
            return niveau, interpretation
    return "—", ""


def normalize_signature(indicateur: str, value: float) -> float:
    """Normalise la signature stylométrique sur une échelle [0,1] stable par axe."""
    bornes: dict[str, tuple[float, float]] = {
        "Noms & adjectifs": (0.0, 0.60),
        "Verbes d'action": (0.0, 0.40),
        "Nuances (adverbes)": (0.0, 0.30),
        "Ponctuation": (0.0, 0.35),
        "Longueur des mots": (3.0, 10.0),
        "Participes vs conjugués": (0.0, 1.0),
        "Déterminants définis": (0.0, 1.0),
    }
    mn, mx = bornes.get(indicateur, (0.0, 1.0))
    if mx - mn < 1e-6:
        return 0.0
    return min(1.0, max(0.0, (value - mn) / (mx - mn)))


def coherence_level(score: int) -> tuple[str, str]:
    """Retourne (label, tonalité streamlit) pour un score de cohérence."""
    if score >= 80:
        return "Excellent", "success"
    if score >= 65:
        return "Bon", "info"
    if score >= 45:
        return "À surveiller", "warning"
    return "Critique", "error"


def compute_coherence_score(
    sig_fiche: dict[str, float], sig_dataset: dict[str, float], mots_repetes: list[str]
) -> tuple[int, dict[str, float]]:
    """Calcule un score global de cohérence (0-100) à partir des écarts stylométriques."""
    deltas: dict[str, float] = {}
    for k in sig_fiche:
        nf = normalize_signature(k, sig_fiche[k])
        nd = normalize_signature(k, sig_dataset[k])
        deltas[k] = abs(nf - nd)
    avg_delta = sum(deltas.values()) / max(1, len(deltas))
    base_score = 100 * (1 - avg_delta)
    rep_penalty = min(20, max(0, len(mots_repetes) - 1) * 2)
    final_score = int(max(0, min(100, round(base_score - rep_penalty))))
    return final_score, deltas


def prioritized_actions(
    stats: dict, deltas: dict[str, float], max_actions: int = 3
) -> list[str]:
    """Génère des conseils d'écriture concrets à partir des métriques courantes."""
    actions: list[str] = []
    if stats["ratio"] < 1.3:
        actions.append(
            "Ta prose reste très proche du brouillon — essaie d'ajouter des "
            "détails, des images ou des précisions pour enrichir le texte."
        )
    elif stats["ratio"] > 3.0:
        actions.append(
            "Tu développes beaucoup — vérifie que chaque ajout apporte du sens, "
            "sinon élague les passages redondants."
        )

    if stats["ttr"] < 0.50:
        actions.append(
            "Plusieurs mots reviennent souvent — cherche des synonymes ou "
            "reformule pour diversifier le vocabulaire."
        )
    elif stats["ttr"] > 0.85:
        actions.append(
            "Le vocabulaire est très varié — assure-toi que le registre reste "
            "cohérent d'une fiche à l'autre."
        )

    if stats["long_moy_phrases"] < 10:
        actions.append(
            "Tes phrases sont courtes — essaie d'en relier certaines pour "
            "obtenir un rythme plus fluide."
        )
    elif stats["long_moy_phrases"] > 25:
        actions.append(
            "Tes phrases sont longues — découpe-en quelques-unes pour "
            "faciliter la lecture."
        )

    if stats["mots_repetes"]:
        sample = ", ".join(f"« {m} »" for m in stats["mots_repetes"][:3])
        actions.append(
            f"Les mots {sample} reviennent souvent — remplace-en "
            "certains par des synonymes."
        )

    top_deltas = sorted(deltas.items(), key=lambda x: x[1], reverse=True)[:2]
    for axis, delta in top_deltas:
        if delta > 0.35:
            actions.append(
                f"Ta fiche s'écarte de la moyenne du dataset sur « {axis} » — "
                "rapproche-toi du style général pour garder la cohérence."
            )

    dedup = list(dict.fromkeys(actions))
    return dedup[:max_actions]


def translate_trigram(trigram: str) -> str:
    """Traduit un trigramme POS (ex. 'DET-NOUN-VERB') en français lisible."""
    return " · ".join(_POS_FR.get(tag, tag) for tag in trigram.split("-"))


def get_pos_trigrams(text: str, nlp) -> Counter | None:
    """
    Extrait les trigrammes POS d'un texte.
    Returns:
        Counter des trigrammes (ex. "DET-NOUN-VERB") ou None.
    """
    if nlp is None or not text:
        return None
    doc = nlp(text)
    tags = [t.pos_ for t in doc if not t.is_space]
    if len(tags) < 3:
        return None
    trigrams = [f"{tags[i]}-{tags[i + 1]}-{tags[i + 2]}" for i in range(len(tags) - 2)]
    return Counter(trigrams)


def get_stylometric_signature(text: str, nlp) -> dict[str, float] | None:
    """
    Signature stylométrique (ADN stylistique).
    Retourne None si nlp absent ou texte vide.
    """
    if nlp is None or not text:
        return None
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    nb_tokens = max(1, len(tokens))
    counts = Counter(t.pos_ for t in doc)
    nb_punct = len([t for t in doc if t.is_punct])

    verbs = [t for t in doc if t.pos_ in ("VERB", "AUX")]
    participes = sum(1 for t in verbs if "Part" in t.morph.get("VerbForm", []))
    conjugues = sum(1 for t in verbs if "Fin" in t.morph.get("VerbForm", []))
    ratio_part = participes / max(1, participes + conjugues)

    dets = [t for t in doc if t.pos_ == "DET"]
    dets_def = sum(1 for t in dets if "Def" in t.morph.get("Definite", []))
    dets_indef = sum(1 for t in dets if "Ind" in t.morph.get("Definite", []))
    ratio_def = dets_def / max(1, dets_def + dets_indef)

    return {
        "Noms & adjectifs": (counts.get("NOUN", 0) + counts.get("ADJ", 0)) / nb_tokens,
        "Verbes d'action": counts.get("VERB", 0) / nb_tokens,
        "Nuances (adverbes)": counts.get("ADV", 0) / nb_tokens,
        "Ponctuation": nb_punct / nb_tokens,
        "Longueur des mots": sum(len(t.text) for t in tokens) / nb_tokens,
        "Participes vs conjugués": ratio_part,
        "Déterminants définis": ratio_def,
    }


def compute_row_cache(
    edit_input: str,
    edit_output: str,
    nlp,
    df_valid: pd.DataFrame,
    row_id: str,
    cache_columns: list[str],
    get_avg_signature: Callable[[pd.DataFrame], dict[str, float] | None],
) -> dict[str, str]:
    """
    Calcule les valeurs de cache pour une seule ligne (sauvegarde).
    N'appelle spaCy que sur cette ligne. Retourne un dict colonne -> valeur string.
    """
    if nlp is None or not (edit_input and edit_output):
        return {c: "" for c in cache_columns}
    ins = get_linguistic_insights(edit_input, edit_output, nlp)
    sig_fiche = get_stylometric_signature(edit_output, nlp)
    tri = get_pos_trigrams(edit_output, nlp)
    if not ins or not sig_fiche:
        return {c: "" for c in cache_columns}

    others = df_valid[df_valid["id"].astype(str) != str(row_id)]
    sig_dataset = get_avg_signature(others)
    if sig_dataset:
        score, _ = compute_coherence_score(
            sig_fiche, sig_dataset, ins.get("mots_repetes", [])
        )
    else:
        score = 100

    return {
        "_ratio": str(round(ins["ratio"], 3)),
        "_ttr": f"{ins['ttr']:.2f}",
        "_long_phrases": str(round(ins["long_moy_phrases"], 1)),
        "_signature_json": json.dumps(sig_fiche),
        "_coherence_score": str(score),
        "_trigrams_json": json.dumps(dict(tri)) if tri else "{}",
    }


def signature_variance(df_valid: pd.DataFrame) -> dict[str, float] | None:
    """Calcule l'écart-type de chaque axe stylistique depuis le cache _signature_json.

    Complément orthogonal de avg_signature_from_cache (même données, autre statistique).
    Un écart élevé signale un dataset hétérogène sur cet axe.

    Args:
        df_valid: Sous-ensemble du DataFrame filtré sur STATUT_VALIDE.

    Returns:
        Dict {axe: écart-type} avec les mêmes clés que get_stylometric_signature,
        ou None si moins de 2 signatures disponibles.
    """
    sigs: list[dict[str, float]] = []
    for _, row in df_valid.iterrows():
        raw = row.get("_signature_json", "") or ""
        if not raw:
            continue
        try:
            sigs.append(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            continue
    if len(sigs) < 2:
        return None
    keys = list(sigs[0].keys())
    result: dict[str, float] = {}
    for k in keys:
        values = [s[k] for s in sigs if k in s]
        if not values:
            result[k] = 0.0
            continue
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        result[k] = round(variance ** 0.5, 4)
    return result
