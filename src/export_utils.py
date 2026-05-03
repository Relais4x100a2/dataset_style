"""
Export multi-modèles : JSONL pour fine-tuning (LFM2, Baguettotron, Mistral).
"""

import io
import json
from typing import Literal

import pandas as pd

from src.database import STATUT_VALIDE

ExportFormat = Literal["lfm2", "baguettotron", "mistral"]


def _stylometry_summary(row: pd.Series) -> str:
    """
    Construit une ligne compacte d'indicateurs stylométriques à partir des colonnes cache.
    Utilisée pour injection optionnelle dans les exports (system/user ou trace).
    """
    parts = []
    ttr = row.get("_ttr")
    if ttr is not None and str(ttr).strip():
        try:
            parts.append(f"TTR≈{float(ttr):.2f}")
        except (ValueError, TypeError):
            pass
    long_ph = row.get("_long_phrases")
    if long_ph is not None and str(long_ph).strip():
        try:
            parts.append(f"{float(long_ph):.0f} mots/phrase")
        except (ValueError, TypeError):
            pass
    dens = row.get("_lexical_density")
    if dens is not None and str(dens).strip():
        try:
            parts.append(f"densité {float(dens):.2f}")
        except (ValueError, TypeError):
            pass
    nb_sent = row.get("_nb_sentences")
    if nb_sent is not None and str(nb_sent).strip():
        try:
            parts.append(f"{int(float(nb_sent))} phrases")
        except (ValueError, TypeError):
            pass
    weak = row.get("_weak_verb_ratio")
    if weak is not None and str(weak).strip():
        try:
            parts.append(f"verbes_faibles≈{float(weak):.2f}")
        except (ValueError, TypeError):
            pass
    if not parts:
        return ""
    return " | ".join(parts)


def _instruction_text(row: pd.Series) -> str:
    """Texte d'instruction commun pour les exports."""
    structure = row.get("structure") or row.get("forme") or ""
    format_ = row.get("format") or row.get("support") or ""
    public = row.get("public") or ""
    return (
        f"Réécris ce brouillon. Structure textuelle : {structure}. "
        f"Tonalité textuelle : {row['ton']}. Format de sortie : {format_}. "
        f"Public cible : {public}."
    )


def _build_lfm2_messages(row: pd.Series, include_stylometry: bool) -> list[dict]:
    """Construit la liste messages pour LFM2 puis on sérialise en ChatML."""
    instruction = _instruction_text(row)
    user_content = f"{instruction}\n\nBrouillon :\n{row['input']}"
    messages = []
    if include_stylometry:
        stylo = _stylometry_summary(row)
        if stylo:
            system_content = (
                f"Paramètres : type={row['type']}, structure={row.get('structure', '')}, "
                f"ton={row['ton']}, format={row.get('format', '')}, "
                f"public={row.get('public', '')}. "
                f"Indicateurs : {stylo}."
            )
            messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": str(row["output"])})
    return messages


def _convert_to_lfm2_jsonl(df_valid: pd.DataFrame, include_stylometry: bool) -> str:
    """
    Format LFM2-24B-A2B : une entrée = messages (system optionnel, user, assistant).
    Une ligne JSONL par fiche avec clé "messages" (template ChatML appliqué par le pipeline).
    """
    buf = io.StringIO()
    for _, row in df_valid.iterrows():
        messages = _build_lfm2_messages(row, include_stylometry)
        entry = {"messages": messages}
        buf.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return buf.getvalue()


def _convert_to_baguettotron_jsonl(df_valid: pd.DataFrame, include_stylometry: bool) -> str:
    """
    Format PleIAs/Baguettotron : ChatML + <think> trace + <H≈…>.
    Si include_stylometry, ajoute une ligne Stylo dans la trace.
    """
    buf = io.StringIO()
    for _, row in df_valid.iterrows():
        h_token = "<H≈0.3>" if row["type"] == "Normalisation" else "<H≈1.5>"
        short_input = " ".join(str(row.get("input", "")).split()[:5]) + "..."
        trace = f"{row.get('structure', '')} → {row['ton']} ※ {short_input} ∴ {row['type']}"
        if include_stylometry:
            stylo = _stylometry_summary(row)
            if stylo:
                trace += f"\nStylo: {stylo}"
        instruction = _instruction_text(row)
        prompt = (
            f"<|im_start|>user\n{instruction}\n\n"
            f"Brouillon : {row['input']}<|im_end|>\n<|im_start|>assistant"
        )
        response = f"<think>\n{trace}\n</think>\n{h_token} {row['output']}<|im_end|>"
        entry = {"text": f"{prompt}{response}"}
        buf.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return buf.getvalue()


def _convert_to_mistral_jsonl(df_valid: pd.DataFrame, include_stylometry: bool) -> str:
    """
    Format Mistral Small Creative : messages (user, assistant), une ligne JSON par fiche.
    Pas de tokens ChatML dans le contenu ; optionnellement style + stylométrie en préfixe user.
    """
    buf = io.StringIO()
    for _, row in df_valid.iterrows():
        instruction = _instruction_text(row)
        user_content = f"{instruction}\n\nBrouillon :\n{row['input']}"
        if include_stylometry:
            stylo = _stylometry_summary(row)
            if stylo:
                user_content = f"[Indicateurs stylométriques : {stylo}]\n\n{user_content}"
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": str(row["output"])},
        ]
        entry = {"messages": messages}
        buf.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return buf.getvalue()


def convert_to_jsonl(
    df: pd.DataFrame,
    format: ExportFormat,
    include_stylometry: bool = False,
) -> str:
    """
    Point d'entrée unique : exporte le dataset (fiches « Fait et validé ») en JSONL
    selon le format cible (LFM2, Baguettotron, Mistral).
    """
    df_valid = df[df["statut"] == STATUT_VALIDE]
    if format == "lfm2":
        return _convert_to_lfm2_jsonl(df_valid, include_stylometry)
    if format == "baguettotron":
        return _convert_to_baguettotron_jsonl(df_valid, include_stylometry)
    if format == "mistral":
        return _convert_to_mistral_jsonl(df_valid, include_stylometry)
    raise ValueError(f"Format inconnu : {format}")
