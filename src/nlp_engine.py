"""
Moteur linguistique : insights linguistiques, stylométrie, cohérence.
Sans dépendance Streamlit — testable indépendamment.
"""

import json
import logging
import os
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests

logger = logging.getLogger(__name__)

LANGUAGETOOL_PUBLIC_URL = "https://api.languagetool.org/v2/check"
LANGUAGETOOL_TIMEOUT = 15


def languagetool_check_url(base_url: str | None = None) -> str:
    """URL complète du endpoint ``/v2/check`` (serveur local ou API publique)."""
    base = (base_url or os.environ.get("LANGUAGETOOL_BASE_URL") or "").strip().rstrip("/")
    if not base:
        return LANGUAGETOOL_PUBLIC_URL
    return urljoin(base + "/", "v2/check")


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


def corriger_texte_fr(text: str, languagetool_base_url: str | None = None) -> str:
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
            languagetool_check_url(languagetool_base_url),
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
    comptage = Counter(t.lemma_.lower() for t in doc_out if not t.is_punct and not t.is_stop)
    mots_repetes = [
        lem for lem, n in comptage.items() if n >= seuil_repetition and lem and str(lem).strip()
    ]

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
    verb_counts = Counter(t.lemma_.lower() for t in doc if t.pos_ in ("VERB", "AUX"))
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
        total += abs(sig_in.get(k, 0) - sig_out.get(k, 0))
    return min(1.0, total / max(1, len(keys)) * 2)


# Table de données des paliers : indicateur → [(seuil_max, niveau, interprétation), ...]
# Le dernier tuple (seuil_max=inf) est le palier par défaut (valeur >= dernier seuil).
_PALIERS: dict[str, list[tuple[float, str, str]]] = {
    "ratio": [
        (1.3, "Minimal", "Tu restes proche du brouillon."),
        (2.0, "Progressif", "Tu développes."),
        (2.5, "Solide", "Tu as bien développé l'idée."),
        (float("inf"), "Amplifié", "Tu déploies beaucoup."),
    ],
    "ttr": [
        (0.50, "Bas", "Vocabulaire répétitif."),
        (0.65, "Intermédiaire", "Vocabulaire correct."),
        (0.80, "Élevé", "Vocabulaire soutenu."),
        (float("inf"), "Très élevé", "Vocabulaire très riche."),
    ],
    "moy_phrases": [
        (10, "Court", "Rythme vif, phrases courtes."),
        (18, "Équilibré", "Rythme équilibré."),
        (25, "Ample", "Rythme ample."),
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


COHERENCE_SCORE_MIN: int = 0
COHERENCE_SCORE_MAX: int = 100

CURATOR_MESSAGE_STATS_UNAVAILABLE: str = (
    "Analyse stylométrique indisponible : modèle NLP absent ou textes insuffisants "
    "pour calculer les indicateurs."
)

CURATOR_MESSAGE_ADVICE_BALANCED: str = (
    "Aucun conseil prioritaire : indicateurs dans une zone équilibrée, ou dataset "
    "encore trop réduit pour comparer les axes stylistiques."
)


def qualitative_coherence_feedback(score: int | None) -> tuple[str, str]:
    """Libellé qualitatif et tonalité UI pour le score persisté.

    ``coherence_level`` n'est appelé que pour un entier dans la plage métier
    ``[COHERENCE_SCORE_MIN, COHERENCE_SCORE_MAX]``. Valeur absente ou hors plage :
    message explicite en français, sans palier inventé.

    Args:
        score: Score issu du cache persisté, ou ``None`` si absent / non numérique.

    Returns:
        Couple ``(libellé, tonalité)`` pour ``st.success`` / ``st.warning`` / etc.
    """
    if score is None:
        return "Non calculé", "warning"
    if score < COHERENCE_SCORE_MIN or score > COHERENCE_SCORE_MAX:
        return (
            "Score de cohérence absent, illisible ou hors plage (attendu 0–100).",
            "warning",
        )
    return coherence_level(score)


def normalized_coherence_metric_score(score: int | None) -> int | None:
    """Score prêt pour ``st.metric`` : ``None`` si absent ou hors plage 0–100."""
    if score is None:
        return None
    if score < COHERENCE_SCORE_MIN or score > COHERENCE_SCORE_MAX:
        return None
    return int(score)


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
    stats: dict[str, Any], deltas: dict[str, float], max_actions: int = 3
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
            "Tes phrases sont longues — découpe-en quelques-unes pour faciliter la lecture."
        )

    if stats["mots_repetes"]:
        sample = ", ".join(f"« {m} »" for m in stats["mots_repetes"][:3])
        actions.append(
            f"Les mots {sample} reviennent souvent — remplace-en certains par des synonymes."
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


def curator_advices_after_save(stats: dict[str, Any], deltas: dict[str, float]) -> list[str]:
    """Conseils pour l'UI après sauvegarde ; message explicite si NLP ou stats indisponibles."""
    if not stats:
        return [CURATOR_MESSAGE_STATS_UNAVAILABLE]
    adv = prioritized_actions(stats, deltas)
    if adv:
        return adv
    return [CURATOR_MESSAGE_ADVICE_BALANCED]


def avg_signature_from_cache(df: pd.DataFrame) -> dict[str, float] | None:
    """Moyenne des signatures stylométriques persistées (_signature_json), lignes non vides."""
    sigs: list[dict[str, float]] = []
    if df is None or df.empty:
        return None
    for _, row in df.iterrows():
        raw = str(row.get("_signature_json", "") or "").strip()
        if not raw:
            continue
        try:
            sigs.append(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            continue
    if not sigs:
        return None
    keys = list(sigs[0].keys())
    n = len(sigs)
    return {k: sum(s.get(k, 0.0) for s in sigs) / n for k in keys}


@dataclass(frozen=True)
class RowNlpCacheResult:
    """Colonnes de cache NLP persistées + données pour le feedback curateur (post-sauvegarde)."""

    cache: dict[str, str]
    coherence_score: int | None
    coherence_deltas: dict[str, float]
    advice_stats: dict[str, Any]


def post_save_stylometric_session_payload(pkg: RowNlpCacheResult) -> dict[str, Any]:
    """Construit le dict de session Streamlit pour le bloc « retour stylistique » post-sauvegarde.

    Les métriques TTR et contraste syntaxique proviennent des colonnes cache de ``pkg``
    (ligne relue après persistance). Le score affiché en métrique n'est renseigné que
    s'il est dans l'intervalle métier 0–100 ; le libellé qualitatif évite ``coherence_level``
    hors plage. Les conseils passent par :func:`curator_advices_after_save` (liste jamais vide).

    Args:
        pkg: Bundle produit par :func:`row_nlp_feedback_bundle_after_persist`.

    Returns:
        Dictionnaire sérialisable (``score``, ``ttr``, ``contrast``, ``level``, ``tone``,
        ``advices``) pour ``st.session_state``.
    """
    advices = curator_advices_after_save(pkg.advice_stats, pkg.coherence_deltas)
    raw_score = pkg.coherence_score
    ttr = (pkg.cache.get("_ttr") or "").strip() or "—"
    contrast = (pkg.cache.get("_syntax_contrast") or "").strip() or "—"
    level, tone = qualitative_coherence_feedback(raw_score)
    metric_score = normalized_coherence_metric_score(raw_score)
    return {
        "score": metric_score,
        "ttr": ttr,
        "contrast": contrast,
        "level": level,
        "tone": tone,
        "advices": advices,
    }


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


def _get_finetuning_insights(text_in: str, text_out: str, nlp) -> dict[str, str] | None:
    """
    Calcule les indicateurs additionnels pour le fine-tuning (densité lexicale,
    verbes faibles, contraste syntaxique, nb phrases, ponctuation expressive, stop ratio).
    Retourne un dict colonne -> valeur string, ou None si nlp/texte manquant.
    """
    if nlp is None or not (text_in and text_out):
        return None
    doc_out = nlp(text_out)
    tokens = [t for t in doc_out if not t.is_punct and not t.is_space]
    nb_tokens = max(1, len(tokens))
    content_pos = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
    content_count = sum(1 for t in doc_out if not t.is_punct and t.pos_ in content_pos)
    lexical_density = content_count / nb_tokens

    verb_counts = Counter(t.lemma_.lower() for t in doc_out if t.pos_ in ("VERB", "AUX"))
    total_verbs = sum(verb_counts.values())
    weak_count = sum(verb_counts.get(v, 0) for v in VERBES_FAIBLES)
    weak_verb_ratio = weak_count / max(1, total_verbs)

    contrast = syntax_contrast_score(text_in, text_out, nlp)
    nb_sentences = len(list(doc_out.sents))

    punct_tiret = text_out.count("—")
    punct_ellipsis = text_out.count("...")
    punct_colon = text_out.count(":")
    punct_exp = f"{punct_tiret},{punct_ellipsis},{punct_colon}"

    stop_out = sum(1 for t in doc_out if not t.is_punct and t.is_stop)
    stop_ratio_out = stop_out / nb_tokens

    return {
        "_lexical_density": f"{lexical_density:.2f}",
        "_weak_verb_ratio": f"{weak_verb_ratio:.2f}",
        "_syntax_contrast": f"{contrast:.2f}",
        "_nb_sentences": str(nb_sentences),
        "_punct_exp": punct_exp,
        "_stop_ratio_out": f"{stop_ratio_out:.2f}",
    }


def compute_row_cache(
    edit_input: str,
    edit_output: str,
    nlp,
    df_valid: pd.DataFrame,
    row_id: str,
    cache_columns: list[str],
    get_avg_signature: Callable[[pd.DataFrame], dict[str, float] | None],
) -> RowNlpCacheResult:
    """
    Calcule les valeurs de cache pour une seule ligne (sauvegarde).

    N'appelle spaCy que sur cette ligne. Retourne les colonnes persistables ainsi que
    score, deltas et stats alignés avec ``prioritized_actions`` (même passage NLP).
    """
    empty = {c: "" for c in cache_columns}
    if nlp is None or not (edit_input and edit_output):
        return RowNlpCacheResult(empty, None, {}, {})
    ins = get_linguistic_insights(edit_input, edit_output, nlp)
    sig_fiche = get_stylometric_signature(edit_output, nlp)
    tri = get_pos_trigrams(edit_output, nlp)
    if not ins or not sig_fiche:
        return RowNlpCacheResult(empty, None, {}, {})

    advice_stats: dict[str, Any] = {
        "ratio": ins["ratio"],
        "ttr": ins["ttr"],
        "long_moy_phrases": ins["long_moy_phrases"],
        "mots_repetes": ins.get("mots_repetes", []),
    }

    others = df_valid[df_valid["id"].astype(str) != str(row_id)]
    sig_dataset = get_avg_signature(others)
    deltas: dict[str, float]
    if sig_dataset:
        score, deltas = compute_coherence_score(sig_fiche, sig_dataset, ins.get("mots_repetes", []))
    else:
        score, deltas = 100, {}

    result = {
        "_ratio": str(round(ins["ratio"], 3)),
        "_ttr": f"{ins['ttr']:.2f}",
        "_long_phrases": str(round(ins["long_moy_phrases"], 1)),
        "_signature_json": json.dumps(sig_fiche),
        "_coherence_score": str(score),
        "_trigrams_json": json.dumps(dict(tri)) if tri else "{}",
    }
    extra = _get_finetuning_insights(edit_input, edit_output, nlp)
    if extra:
        result.update(extra)
    for col in cache_columns:
        if col not in result:
            result[col] = ""
    return RowNlpCacheResult(result, score, deltas, advice_stats)


def _parse_cache_float(value: object, default: float = 0.0) -> float:
    """Parse une valeur numérique issue du cache persisté (chaîne SQL / CSV)."""
    s = str(value or "").strip().replace(",", ".")
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def _parse_cache_int_score(value: object) -> int | None:
    """Parse ``_coherence_score`` persisté ; ``None`` si absent ou invalide."""
    s = str(value or "").strip()
    if not s:
        return None
    try:
        return int(round(float(s.replace(",", "."))))
    except ValueError:
        return None


def parse_persisted_coherence_score(value: object) -> int | None:
    """Parse une cellule ``_coherence_score`` persistée (mêmes règles que le cache SQL).

    Exposé pour le tableau de bord et tout agrégat hors moteur de sauvegarde.

    Args:
        value: Valeur brute (souvent ``str``) lue depuis le DataFrame.

    Returns:
        Entier 0–100 arrondi, ou ``None`` si absent ou non numérique.
    """
    return _parse_cache_int_score(value)


def parse_persisted_syntax_contrast(value: object) -> float | None:
    """Parse une cellule ``_syntax_contrast`` ; ``None`` si vide ou invalide.

    Contrairement à ``_parse_cache_float`` (défaut 0.0), les cellules vides sont
    exclues des moyennes du tableau de bord.

    Args:
        value: Chaîne ou valeur SQL/CSV.

    Returns:
        Flottant typiquement dans ``[0, 1]``, ou ``None``.
    """
    s = str(value or "").strip().replace(",", ".")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def list_parsed_coherence_scores(
    df: pd.DataFrame,
    column: str = "_coherence_score",
) -> list[int]:
    """Liste les scores de cohérence numériques d'un sous-ensemble de lignes."""
    if column not in df.columns or df.empty:
        return []
    out: list[int] = []
    for v in df[column].tolist():
        p = parse_persisted_coherence_score(v)
        if p is not None:
            out.append(p)
    return out


def mean_syntax_contrast_parsed(
    df: pd.DataFrame,
    column: str = "_syntax_contrast",
) -> float | None:
    """Moyenne du contraste syntaxique sur les cellules parseables uniquement."""
    if column not in df.columns or df.empty:
        return None
    vals: list[float] = []
    for v in df[column].tolist():
        p = parse_persisted_syntax_contrast(v)
        if p is not None:
            vals.append(p)
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def dataframe_for_dashboard_scope(
    df: pd.DataFrame,
    *,
    validated_only: bool,
    validated_label: str,
    statut_column: str = "statut",
) -> pd.DataFrame:
    """Filtre le DataFrame selon le périmètre choisi en UI (validées ou tout)."""
    if df.empty:
        return df.copy()
    if not validated_only:
        return df.copy()
    if statut_column not in df.columns:
        return df.iloc[0:0].copy()
    mask = df[statut_column].astype(str) == str(validated_label)
    return df.loc[mask].copy()


def coherence_score_bucket_table(scores: list[int]) -> pd.DataFrame:
    """Comptages par tranches de 10 points pour histogramme (0–100)."""
    labels = [
        "0–9",
        "10–19",
        "20–29",
        "30–39",
        "40–49",
        "50–59",
        "60–69",
        "70–79",
        "80–89",
        "90–100",
    ]
    rows: list[dict[str, object]] = []
    for i, lab in enumerate(labels):
        lo = i * 10
        hi = lo + 9
        if i == 9:
            hi = 100
        cnt = sum(1 for s in scores if lo <= s <= hi)
        rows.append({"Tranche (score)": lab, "Nombre": int(cnt)})
    out_of_range = sum(1 for s in scores if s < 0 or s > 100)
    if out_of_range:
        rows.append({"Tranche (score)": "Hors plage", "Nombre": int(out_of_range)})
    return pd.DataFrame(rows)


def outliers_low_coherence_table(
    df: pd.DataFrame,
    *,
    limit: int = 15,
    score_column: str = "_coherence_score",
) -> pd.DataFrame:
    """Sous-table triée : entrées au score de cohérence le plus bas (parseable)."""
    if df.empty or score_column not in df.columns:
        return pd.DataFrame()
    parsed: list[tuple[int, int]] = []
    for idx, v in enumerate(df[score_column].tolist()):
        p = parse_persisted_coherence_score(v)
        if p is not None:
            parsed.append((idx, p))
    if not parsed:
        return pd.DataFrame()
    parsed.sort(key=lambda t: (t[1], str(df.iloc[t[0]].get("id", ""))))
    take = parsed[: max(1, min(limit, len(parsed)))]
    idxs = [i for i, _ in take]
    part = df.iloc[idxs].copy()
    part["score_coherence"] = [sc for _, sc in take]
    cols = [c for c in ("id", "statut", "type", "score_coherence") if c in part.columns]
    return part[cols].reset_index(drop=True)


def row_nlp_feedback_bundle_after_persist(
    df_entries: pd.DataFrame,
    row_id: str,
    nlp,
    cache_columns: list[str],
) -> RowNlpCacheResult:
    """Construit le bundle feedback à partir des entrées rechargées après commit SQL.

    Le score de cohérence, le TTR et le contraste syntaxique affichés en UI doivent
    refléter les colonnes de cache lues depuis la base (relecture ``load_project_entries``).
    Les deltas pour ``prioritized_actions`` sont recalculés à partir des signatures
    JSON persistées et de la moyenne des autres lignes ; ``mots_repetes`` repasse par
    ``get_linguistic_insights`` lorsque spaCy est disponible, pour rester aligné avec
    le même type d'analyse qu'à la sauvegarde.

    Args:
        df_entries: DataFrame projet tel que renvoyé après persistance (ex. SQL).
        row_id: Identifiant stable de la ligne enregistrée.
        nlp: Pipeline spaCy ou ``None``.
        cache_columns: Liste des colonnes de cache attendues (ex. ``CACHE_COLUMNS``).

    Returns:
        Bundle prêt pour ``curator_advices_after_save`` / affichage métrique.
    """
    empty_cache = {c: "" for c in cache_columns}
    rid = str(row_id)
    if df_entries is None or df_entries.empty:
        return RowNlpCacheResult(empty_cache, None, {}, {})

    matches = df_entries[df_entries["id"].astype(str) == rid]
    if matches.empty:
        return RowNlpCacheResult(empty_cache, None, {}, {})

    row = matches.iloc[0]
    cache: dict[str, str] = {col: str(row.get(col, "") or "") for col in cache_columns}

    persisted_score = _parse_cache_int_score(cache.get("_coherence_score", ""))

    raw_sig = str(cache.get("_signature_json", "") or "").strip()
    sig_fiche: dict[str, float] | None = None
    if raw_sig:
        try:
            loaded = json.loads(raw_sig)
            if isinstance(loaded, dict):
                sig_fiche = {
                    str(k): float(v) for k, v in loaded.items() if isinstance(v, (int, float))
                }
        except (json.JSONDecodeError, TypeError, ValueError):
            sig_fiche = None

    others = df_entries[df_entries["id"].astype(str) != rid]
    sig_dataset = avg_signature_from_cache(others)

    inp = str(row.get("input", "") or "")
    out = str(row.get("output", "") or "")
    mots_repetes: list[str] = []
    if nlp is not None and inp.strip() and out.strip():
        ins = get_linguistic_insights(inp, out, nlp)
        if ins:
            mots_repetes = list(ins.get("mots_repetes") or [])

    deltas: dict[str, float] = {}
    if sig_fiche and sig_dataset:
        _, deltas = compute_coherence_score(sig_fiche, sig_dataset, mots_repetes)

    ratio_s = str(cache.get("_ratio", "") or "").strip()
    ttr_s = str(cache.get("_ttr", "") or "").strip()
    long_s = str(cache.get("_long_phrases", "") or "").strip()
    has_numeric_cache = bool(ratio_s and ttr_s and long_s)

    advice_stats: dict[str, Any] = {}
    if has_numeric_cache:
        advice_stats = {
            "ratio": _parse_cache_float(cache.get("_ratio")),
            "ttr": _parse_cache_float(cache.get("_ttr")),
            "long_moy_phrases": _parse_cache_float(cache.get("_long_phrases")),
            "mots_repetes": mots_repetes,
        }

    return RowNlpCacheResult(cache, persisted_score, deltas, advice_stats)


def recompute_cache_for_rows(
    df_valid: pd.DataFrame,
    nlp,
    cache_columns: list[str],
) -> pd.DataFrame:
    """
    Recalcule tout le cache stylométrique pour les lignes de df_valid (fiches validées).
    Utilise un passage en deux temps pour la cohérence (moyenne des autres fiches).
    Retourne une copie de df_valid avec les colonnes cache_columns mises à jour.
    """
    if nlp is None or df_valid.empty:
        out = df_valid.copy()
        for c in cache_columns:
            if c not in out.columns:
                out[c] = ""
        return out

    n = len(df_valid)
    row_data: list[dict] = []

    for _, row in df_valid.iterrows():
        inp = str(row.get("input") or "").strip()
        out_text = (str(row.get("output") or "")).strip()
        row_id = row.get("id", "")
        if not (inp and out_text):
            row_data.append({"row_id": row_id, "cache": {col: "" for col in cache_columns}})
            continue
        ins = get_linguistic_insights(inp, out_text, nlp)
        sig = get_stylometric_signature(out_text, nlp)
        tri = get_pos_trigrams(out_text, nlp)
        extra = _get_finetuning_insights(inp, out_text, nlp)
        if not ins or not sig:
            row_data.append({"row_id": row_id, "cache": {col: "" for col in cache_columns}})
            continue
        row_data.append(
            {
                "row_id": row_id,
                "ins": ins,
                "sig": sig,
                "tri": tri,
                "extra": extra or {},
            }
        )

    all_sigs = [r["sig"] for r in row_data if "sig" in r and r.get("sig")]
    if not all_sigs:
        result = df_valid.copy()
        for c in cache_columns:
            if c not in result.columns:
                result[c] = ""
        return result

    sum_sigs: dict[str, float] = {}
    for k in all_sigs[0]:
        sum_sigs[k] = sum(s[k] for s in all_sigs)

    result = df_valid.copy()
    for c in cache_columns:
        if c not in result.columns:
            result[c] = ""

    for r in row_data:
        row_id = r["row_id"]
        if "cache" in r:
            for col, val in r["cache"].items():
                result.loc[result["id"].astype(str) == str(row_id), col] = val
            continue
        sig_fiche = r["sig"]
        others_mean = {k: (sum_sigs[k] - sig_fiche[k]) / max(1, n - 1) for k in sig_fiche.keys()}
        score, _ = compute_coherence_score(sig_fiche, others_mean, r["ins"].get("mots_repetes", []))
        cache = {
            "_ratio": str(round(r["ins"]["ratio"], 3)),
            "_ttr": f"{r['ins']['ttr']:.2f}",
            "_long_phrases": str(round(r["ins"]["long_moy_phrases"], 1)),
            "_signature_json": json.dumps(sig_fiche),
            "_coherence_score": str(score),
            "_trigrams_json": json.dumps(dict(r["tri"])) if r["tri"] else "{}",
        }
        cache.update(r.get("extra", {}))
        for col in cache_columns:
            cache.setdefault(col, "")
        for col, val in cache.items():
            result.loc[result["id"].astype(str) == str(row_id), col] = val

    return result


def signature_variance(df_valid: pd.DataFrame) -> dict[str, float] | None:
    """Calcule l'écart-type de chaque axe stylistique depuis le cache _signature_json.

    Complément orthogonal de ``avg_signature_from_cache`` (même données, autre statistique).
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
            loaded = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(loaded, dict):
            continue
        sig = {str(k): float(v) for k, v in loaded.items() if isinstance(v, (int, float))}
        if not sig:
            continue
        sigs.append(sig)
    if len(sigs) < 2:
        return None
    keys = list(sigs[0].keys())
    result: dict[str, float] = {}
    for k in keys:
        values = [float(s[k]) for s in sigs if k in s and isinstance(s.get(k), (int, float))]
        if not values:
            result[k] = 0.0
            continue
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        result[k] = round(variance**0.5, 4)
    return result
