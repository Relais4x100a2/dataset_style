# Stylométrie et fine-tuning

Ce document historique décrivait le cache stylométrique de l'application.

Dans la version multi-tenant, les colonnes de cache sont conservées sur la table `entries`:

- `_ratio`
- `_ttr`
- `_long_phrases`
- `_signature_json`
- `_coherence_score`
- `_trigrams_json`
- `_lexical_density`
- `_weak_verb_ratio`
- `_syntax_contrast`
- `_nb_sentences`
- `_punct_exp`
- `_stop_ratio_out`

Ces indicateurs peuvent être exploités lors des exports JSONL.

