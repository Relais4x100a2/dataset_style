"""
Export Baguettotron : format JSONL / ChatML avec <think> et tokens d'entropie.
"""
import io
import json

import pandas as pd


def convert_to_baguettotron_jsonl(df: pd.DataFrame) -> str:
    """
    Convertit le dataset (lignes « Fait et validé ») en JSONL pour fine-tuning
    Baguettotron (ChatML, <think>, <H≈…>).
    """
    jsonl_output = io.StringIO()
    df_valid = df[df["statut"] == "Fait et validé"]
    for _, row in df_valid.iterrows():
        h_token = "<H≈0.3>" if row["type"] == "Normalisation" else "<H≈1.5>"
        short_input = " ".join(str(row.get("input", "")).split()[:5]) + "..."
        trace = f"{row['forme']} → {row['ton']} ※ {short_input} ∴ {row['type']}"
        instruction = (
            f"Réécris ce brouillon. Forme : {row['forme']}. "
            f"Ton : {row['ton']}. Support : {row['support']}."
        )
        prompt = (
            f"<|im_start|>user\n{instruction}\n\n"
            f"Brouillon : {row['input']}<|im_end|>\n<|im_start|>assistant"
        )
        response = f"<think>\n{trace}\n</think>\n{h_token} {row['output']}<|im_end|>"
        entry = {"text": f"{prompt}{response}"}
        jsonl_output.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return jsonl_output.getvalue()
