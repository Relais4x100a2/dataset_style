# Baseline UX — scénario critique (issue-020 / GitHub #182)

**Objectif :** figer une baseline **Streamlit** reproductible pour comparer la webapp préprod en fin de migration (issue-001 archive de bascule), sans persistance PostgreSQL des événements UX (v1).

## Protocole « une surface à la fois » (obligatoire en coexistence)

Pendant la coexistence dev/staging **Streamlit** vs **webapp**, toute collecte baseline reste **mono-surface par `run_id`** : les jalons émis pour un `run_id` donné proviennent **d’une seule** origine UI avant fin de scénario (`EXP-DL` réussi ou fin explicite). Voir **`docs/migration_parity_matrix.md`** (*Protocole recette : une surface à la fois*) et **`docs/merge_ready_checklist.md`** §6.

## Scénario produit (aligné issue-009)

Parcours **projet → entrée → contrôle → export** :

1. **Projet** : contexte actif dans la barre latérale (flux `SB-CTX` dans `docs/migration_parity_matrix.md`).
2. **Entrée** : persistance d’une fiche via **Nouvelle entrée** (`ENT-NEW-WRITE`) ou **Gestion & édition** (`EDI-SAVE`) après succès `update_project_entries`.
3. **Contrôle** : **testable** — lorsque le récapitulatif qualité pré-export est calculé (même périmètre que les téléchargements) : périmètre `summarize_export_perimeter` + agrégats `build_export_quality_recap` sous **Réglages & Export** (`EXP-SCOPE`). C’est l’équivalent fonctionnel d’une relecture « métier » avant fichier ; ce n’est pas le seul temps d’affichage d’onglet (reruns Streamlit).
4. **Export** : génération des octets CSV + JSONL prêts pour les widgets de téléchargement (`EXP-DL`), sans mesurer uniquement le clic client (non instrumenté en v1).

## Métriques v1 (1 à 3)

| Métrique | Définition | Jalons (codes stables) |
| --- | --- | --- |
| **Temps de parcours** | Écarts `monotonic_ns` entre jalons successifs du même `run_id` | **Streamlit** : `SB-CTX` → `ENT-NEW-WRITE` ou `EDI-SAVE` → `EXP-SCOPE` → `EXP-DL`. **Webapp** : `SB-CTX` → `ENT-NEW-WRITE` ou `EDI-SAVE` → `EXP-DL` (un jalon par requête d’export ; pas de `EXP-SCOPE` — écart matrice issue-015). |
| **Erreurs** | Comptage par `api_error_code` (webapp / contrat) ou `streamlit_category` contrôlée | Événements `kind: ux_error` |
| **Questionnaire** | Perception post-tâche (interne) | Voir `docs/ux_baseline_questionnaire.md` — saisie hors bundle machine (formulaire interne) |

## Rapport synthétique (template revue)

À remplir après campagne **avant / après** (même compte test, deux `run_id` distincts, une surface à la fois). Les temps Δ sont dérivés des JSONL (`monotonic_ns`) ou chronomètre manuel si la collecte fichier est désactivée.

| Panneau / campagne | Surface | `run_id` (8 premiers…) | Δ SB-CTX → entrée (ms) | Δ entrée → EXP-DL (ms) | Erreurs (`ux_error`) | Questionnaire (effort 1–7, confiance 1–7) |
| --- | --- | --- | --- | --- | --- | --- |
| *ex. sprint N* | Streamlit | `ux_a1b2…` | … | … | … | … |
| *ex. sprint N* | Webapp préprod | `ux_c3d4…` | … | … | … | … |

**Limites de lecture** : panel souvent composé de power users ; ne pas extrapoler à l’ensemble des curateurs. Le gap `EXP-SCOPE` webapp impose une comparaison « temps jusqu’au premier export » plutôt qu’au récap Streamlit.

## Cible documentée (nouveau frontal)

- **Parité de jalons** : la webapp émet les mêmes `milestone_code` **hors** `EXP-SCOPE` (non porté par le slice HTTP — récap qualité pré-export : écart documenté matrice issue-015). Implémentation : `src/webapp/ux_telemetry.py` + routes `src/webapp/app.py`.
- **Temps** : mesurer côté client la durée entre réponse `POST`/`PATCH` entrées (persistance réussie) et fin de préparation du fichier export (équivalent `EXP-DL`), ou horodatages serveur (`monotonic_ns` dans les JSONL).
- **Contrôle** : côté webapp v1, pas d’équivalent au récap qualité Streamlit avant export ; le questionnaire et la grille d’erreurs restent la couche « contrôle perception » pour ce gap.

## Webapp (FastAPI)

Prérequis serveur : **`DATASET_STYLE_UX_TELEMETRY_DIR`** défini (sinon aucune écriture JSONL ; les en-têtes de réponse ci-dessous ne sont pas ajoutés).

- **`X-Dataset-Style-Ux-Run-Id`** (requête) : identifiant anonyme stable pour tout le parcours mesuré, même format que Streamlit : `ux_` + **8 à 120** caractères hexadécimaux (ex. `ux_` + UUID sans tirets). Le client doit réutiliser la **même** valeur du début du scénario jusqu’aux exports.
- **`X-Dataset-Style-Ux-Scenario-Id`** (requête, optionnel) : identifiant de scénario versionné ; défaut = `critical_v1_issue_020` (même défaut que `src/ux_scenario_telemetry.py`).
- **Réponses** : lorsque le répertoire télémétrie est actif **et** que la requête porte un `run_id` valide, le serveur renvoie **`X-Dataset-Style-Ux-Run-Id`** (écho) et **`X-Dataset-Style-Ux-Telemetry: 1`** pour corrélation outillage / captures réseau.
- **`SB-CTX`** : `GET /api/projects` après résolution du projet actif (`active_hint` + liste projet) ; **dédupliqué** par couple `(run_id, empreinte projet)` pour limiter le bruit sur les rafraîchissements.
- **`ENT-NEW-WRITE` / `EDI-SAVE`** : après succès de `POST` / `PATCH …/entries` (même `run_id` dans les en-têtes).
- **`EXP-DL`** : une ligne par réponse `GET …/export.csv` ou `GET …/export.jsonl` **réussie** (chaque requête HTTP compte : un testeur qui relance le même export produit un nouveau jalon). Le champ `extra.delivery` vaut `csv` ou `jsonl`. Deux téléchargements (CSV puis JSONL) = **deux** jalons `EXP-DL` (à comparer au parcours Streamlit où un seul rendu d’onglet peut regrouper les deux fichiers).
- **Erreurs** : `ux_error` avec codes `api_errors` si `run_id` présent — export **413** (`EXPORT_PAYLOAD_TOO_LARGE`) ; **404** opaque sur `PATCH …/entries` (entrée absente / refus).

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
