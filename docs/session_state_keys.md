# Clés `session_state` — onglet « Nouvelle entrée »

Référence pour l’issue **017** (texte LLM + isolation multi-utilisateur). Les clés sont construites dans `src/ui_components.py`.

## Préfixe

- **Modèle** : `new_entry_{project_id}_u_{user_id_sanitisé}_*`
- **`user_id_sanitisé`** : identifiant auth stable, normalisé (caractères non sûrs remplacés par `_`, tronqué à 80 caractères ; valeur vide → `anonymous`).

## Clés par champ

| Champ logique | Suffixe de clé Streamlit |
|---------------|-------------------------|
| Brouillon (`text_area`) | `_input` |
| Texte généré (`text_area`) | `_output` |
| Type | `_type` |
| Structure | `_structure` |
| Tonalité | `_ton` |
| Format | `_format` |
| Public | `_public` |
| Statut | `_statut` |
| Notes | `_notes` |

Fonction utilitaire : `new_entry_session_keys(project_id, user_id)`.

## Boutons LLM / enregistrement

- **Générer texte** : `on_click` met à jour la clé `_output` via `commit_new_entry_llm_result`, puis `st.rerun()`.
- **Générer brouillon** : idem pour `_input`.
- **Enregistrer** : lit les mêmes clés ; après succès, drapeau  
  `_pending_clear_new_entry_{project_id}_u_{user_id_sanitisé}`  
  pour vider les tampons au run suivant (avant instanciation des `text_area`).

## Migration (déploiement)

Les anciennes clés **sans** segment utilisateur (`new_entry_{project_id}_input`, etc.) sont copiées une fois vers le nouveau préfixe si les tampons user-scopés sont encore vides, puis supprimées. L’ancien drapeau `_pending_clear_new_entry_{project_id}` est encore reconnu une fois pour déclencher le vidage.
