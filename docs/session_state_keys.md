# Clés `session_state` — onglet « Nouvelle entrée »

Référence pour l’issue **017** (texte LLM + isolation multi-utilisateur). Les clés sont construites dans `src/ui_components.py`.

## Équivalent webapp (issue-010)

Le service **webapp** (FastAPI, page `GET /`) n’utilise pas `st.session_state` : le **projet courant** est stocké dans le **`sessionStorage`** du navigateur sous la clé fixe **`webapp_active_project_id`**. Au chargement des projets, la valeur est renvoyée au serveur comme **`active_hint`** sur `GET /api/projects` pour retrouver le même projet actif que dans la session Streamlit (voir `src/webapp/workspace_payload.py` et `src/project_session.py`).

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

## Migration / héritage (clés sans `user_id`)

Les anciennes clés **sans** segment utilisateur (`new_entry_{project_id}_input`, etc.) ne sont **plus** recopiées vers le compte courant : elles ne portaient pas d’auteur, ce qui permettait à un second compte sur la même session navigateur d’hériter par erreur d’un brouillon. À l’initialisation de l’onglet pour un projet, ces clés sont **supprimées** (tampons vides côté préfixe user-scopé). L’ancien drapeau `_pending_clear_new_entry_{project_id}` reste reconnu dans `render_tab_ajout` pour le vidage (consommé au même titre que le drapeau user-scopé).

**Déconnexion / changement de compte** : `logout` et le chemin `_set_user` lorsque l’`user_id` change appellent `purge_all_new_entry_session_state` (`src/new_entry_session_state.py`) pour effacer **toutes** les clés `new_entry_*` et `_pending_clear_new_entry_*` restées dans `session_state`.
