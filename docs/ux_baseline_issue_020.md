# Baseline UX — scénario critique (issue-020 / GitHub #142)

**Objectif :** figer une baseline **Streamlit** reproductible pour comparer la future webapp en fin de migration (issue-001 archive de bascule), sans persistance PostgreSQL des événements UX (v1).

## Scénario produit (aligné issue-009)

Parcours **projet → entrée → contrôle → export** :

1. **Projet** : contexte actif dans la barre latérale (flux `SB-CTX` dans `docs/migration_parity_matrix.md`).
2. **Entrée** : persistance d’une fiche via **Nouvelle entrée** (`ENT-NEW-WRITE`) ou **Gestion & édition** (`EDI-SAVE`) après succès `update_project_entries`.
3. **Contrôle** : **testable** — lorsque le récapitulatif qualité pré-export est calculé (même périmètre que les téléchargements) : périmètre `summarize_export_perimeter` + agrégats `build_export_quality_recap` sous **Réglages & Export** (`EXP-SCOPE`). C’est l’équivalent fonctionnel d’une relecture « métier » avant fichier ; ce n’est pas le seul temps d’affichage d’onglet (reruns Streamlit).
4. **Export** : génération des octets CSV + JSONL prêts pour les widgets de téléchargement (`EXP-DL`), sans mesurer uniquement le clic client (non instrumenté en v1).

## Métriques v1 (1 à 3)

| Métrique | Définition | Jalons (codes stables) |
| --- | --- | --- |
| **Temps de parcours** | Écarts `monotonic_ns` entre jalons successifs du même `run_id` | `SB-CTX` → `ENT-NEW-WRITE` ou `EDI-SAVE` → `EXP-SCOPE` → `EXP-DL` |
| **Erreurs** | Comptage par `api_error_code` (webapp / contrat) ou `streamlit_category` contrôlée | Événements `kind: ux_error` |
| **Questionnaire** | Perception post-tâche (interne) | Voir `docs/ux_baseline_questionnaire.md` — saisie hors bundle machine (formulaire interne) |

## Cible documentée (nouveau frontal)

- **Parité de jalons** : la webapp devra émettre les **mêmes** `milestone_code` aux frontières équivalentes (session HTTP / client + `run_id` anonyme).
- **Temps** : mesurer côté client la durée entre réponse `POST`/`PATCH` entrées (persistance réussie) et fin de préparation du fichier export (équivalent `EXP-DL`), ou horodatages serveur si instrumentés de façon comparable.
- **Contrôle** : lorsque l’UI webapp expose un récap qualité équivalent au récap Streamlit ; à défaut, documenter un **écart** dans la matrice de parité (comme pour l’export recap slice).

## Collecte technique (interne, sans PII inutile)

- Variable d’environnement **`DATASET_STYLE_UX_TELEMETRY_DIR`** : si définie, append-only **JSONL** journaliers (`ux_scenario_YYYYMMDD.jsonl`, `ux_error_YYYYMMDD.jsonl`).
- Chaque session Streamlit reçoit un `run_id` anonyme (`ux_` + hex) stable pour les reruns.
- **`project_fp`** : 16 premiers caractères hex du SHA-256 du `project_id` (pas d’identifiant brut en fichier).
- Pas de contenu de fiches, pas d’e-mails ; champs `extra` limités à des compteurs / périmètres / tailles d’export.
- Logs applicatifs : chaque événement est aussi émis en `INFO` JSON côté logger (audit central possible sans fichier).

Implémentation : `src/ux_scenario_telemetry.py` ; instrumentation Streamlit : `src/ui_components.py`.

## Cartographie erreurs Streamlit ↔ codes API

| Contexte Streamlit | Code / catégorie enregistré |
| --- | --- |
| Erreur action métier résolue via `resolve_exception_for_api` | `api_error_code` du catalogue (`src/api_errors.py`) |
| Champs obligatoires manquants (Nouvelle entrée) | `streamlit_category: MISSING_REQUIRED_BODY`, `api_error_code` absent |
| Erreurs HTTP LanguageTool / réseau sans mapping | hors périmètre v1 (pas d’événement `ux_error` dédié) |

## Archive issue-001

Consigner le répertoire ou les fichiers JSONL (et éventuellement exports des logs structurés) dans le paquet de revue de bascule. Toute future table d’événements UX devra rester **isolée du schéma tenant** (schéma ou préfixe dédié, rétention) — hors scope v1.
