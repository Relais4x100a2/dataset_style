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

- **Parité de jalons** : la webapp **émet** les mêmes `milestone_code` côté serveur (`src/webapp/ux_telemetry.py` + `src/webapp/app.py`) lorsque le client envoie un `run_id` anonyme ; voir section *Webapp* ci-dessous.
- **Temps** : mesurer côté client la durée entre réponse `POST`/`PATCH` entrées (persistance réussie) et fin de préparation du fichier export (équivalent `EXP-DL`), ou horodatages serveur si instrumentés de façon comparable.
- **Contrôle** : côté webapp, l’équivalent minimal v1 de `EXP-SCOPE` est émis **au moment où le périmètre d’export est résolu** sur le serveur (juste avant la sérialisation CSV/JSONL), **sans** récap qualité détaillé Streamlit — aligné sur l’**écart documenté** matrice (récap pré-export slice).

## Webapp (FastAPI)

- **En-tête obligatoire pour toute écriture fichier** : `X-Dataset-Style-Ux-Run-Id: ux_<32 hex minuscules>` (même format que le `run_id` Streamlit).
- **`SB-CTX`** : sur `GET /api/projects`, ajouter **`X-Dataset-Style-Ux-Shell-Init: 1`** **une seule fois** après connexion (premier chargement shell), avec le `run_id` ci-dessus — évite le bruit sur les polls de liste projets.
- **Jalons automatiques** (si `run_id` valide) : `POST/PATCH …/entries` → `ENT-NEW-WRITE` / `EDI-SAVE` ; `GET …/export.csv` ou `export.jsonl` → enchaînement `EXP-SCOPE` puis `EXP-DL` pour **cette** requête (deux lignes JSONL ; comparer à Streamlit où les deux téléchargements peuvent partager un même rendu d’onglet).
- **Erreur export 413** (`EXPORT_PAYLOAD_TOO_LARGE`) : événement `ux_error` avec `milestone_context: EXP-DL` si `run_id` présent.

Implémentation webapp : `src/webapp/ux_telemetry.py`, routes dans `src/webapp/app.py`.

## Agrégation (revue / issue-001)

- Fichiers produits : `ux_scenario_YYYYMMDD.jsonl`, `ux_error_YYYYMMDD.jsonl` sous le répertoire `DATASET_STYLE_UX_TELEMETRY_DIR`.
- Script TSV (deltas `monotonic_ns` entre jalons successifs par `run_id` + `surface`) : `scripts/aggregate_ux_baseline_jsonl.py --input <répertoire>`.
- Alternative : importer les JSONL dans un tableur ; dédupliquer les doublons éventuels de `SB-CTX` côté webapp en ne conservant que la **première** occurrence par `(run_id, milestone_code)` si besoin.

## Collecte technique (interne, sans PII inutile)

- Variable d’environnement **`DATASET_STYLE_UX_TELEMETRY_DIR`** : si définie, append-only **JSONL** journaliers (`ux_scenario_YYYYMMDD.jsonl`, `ux_error_YYYYMMDD.jsonl`).
- Chaque session Streamlit reçoit un `run_id` anonyme (`ux_` + hex) stable pour les reruns.
- **`project_fp`** : 16 premiers caractères hex du SHA-256 du `project_id` (pas d’identifiant brut en fichier).
- Pas de contenu de fiches, pas d’e-mails ; champs `extra` limités à des compteurs / périmètres / tailles d’export.
- Logs applicatifs : chaque événement est aussi émis en `INFO` JSON côté logger (audit central possible sans fichier).

Implémentation cœur : `src/ux_scenario_telemetry.py` ; instrumentation Streamlit : `src/ui_components.py`.

## Cartographie erreurs Streamlit ↔ codes API

| Contexte Streamlit | Code / catégorie enregistré |
| --- | --- |
| Erreur action métier résolue via `resolve_exception_for_api` | `api_error_code` du catalogue (`src/api_errors.py`) |
| Champs obligatoires manquants (Nouvelle entrée) | `streamlit_category: MISSING_REQUIRED_BODY`, `api_error_code` absent |
| Erreurs HTTP LanguageTool / réseau sans mapping | hors périmètre v1 (pas d’événement `ux_error` dédié) |

## Archive issue-001

Inclure dans le paquet de revue de bascule (issue-001 / stratégie #124) :

1. **Copie** du répertoire configuré (`DATASET_STYLE_UX_TELEMETRY_DIR`) après la campagne de mesure, ou **export** des logs applicatifs contenant les lignes `ux_scenario_event` / `ux_error_event` filtrées sur la période.
2. **Sortie agrégée** : résultat du script `scripts/aggregate_ux_baseline_jsonl.py` (ou tableur équivalent) pour les deltas de jalons.
3. **Questionnaires** papier / formulaire interne : voir `docs/ux_baseline_questionnaire.md` (hors dépôt machine si politique RH l’exige).

Toute future table d’événements UX devra rester **isolée du schéma tenant** (schéma ou préfixe dédié, rétention) — hors scope v1.

Référence positionnement produit : `docs/streamlit_to_new_frontend_cutover.md` (section *Artefacts mesure UX*).
