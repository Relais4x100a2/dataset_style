# Préférences d'affichage (issue-023 / GitHub #186)

Réglages **optionnels** pour la coquille curateur FastAPI (`src/webapp/index_template.py`) : densité de mise en page et confort de lecture. L'expérience par défaut (sans configuration) reste la référence produit.

## Persistance

- **Source de vérité** : colonne PostgreSQL `users.ui_preferences_json` (`TEXT`, défaut `'{}'`), ajoutée par `ensure_schema()` dans `src/database.py` (boot Streamlit et webapp).
- **Plafond** : la charge sérialisée UTF-8 ne doit pas dépasser **4096 octets** (`src/ui_preferences.py`).
- **Premier paint** : un script synchrone en tête du HTML lit `sessionStorage['ds_ui_prefs_v1']` et pose les attributs `data-ds-density` / `data-ds-reading` sur `<html>` avant le rendu principal, afin de limiter le flash entre visites. Le cache est synchronisé après chaque `GET /api/account` ou `PATCH /api/account/ui-preferences` réussi, et effacé à la déconnexion.

## API

| Méthode | Route | Rôle |
|---------|-------|------|
| `GET` | `/api/account` | Inclut `uiPreferences` : `{ "density": "...", "readingComfort": "..." }`. |
| `PATCH` | `/api/account/ui-preferences` | Fusion partielle ; corps JSON avec zéro, une ou deux clés parmi `density`, `readingComfort`. Réponse canonique : `{ "uiPreferences": { ... } }`. |

Valeurs autorisées :

- `density` : `default` | `compact` | `comfortable`
- `readingComfort` : `default` | `high_contrast` | `reduced_motion`

Erreurs de validation : HTTP **400**, enveloppe `{"error": {"code": "BAD_REQUEST", ...}}` (alignement routes webapp existantes).

## Couplage issue-022 (design tokens)

Les attributs **`data-ds-density`** et **`data-ds-reading`** sur `document.documentElement` servent de **point d'ancrage** pour le mapping CSS (`src/webapp/static/design_tokens.css`).

## Sécurité UX (ne pas masquer les alertes)

Les règles CSS de densité / confort ciblent principalement la zone **`.wrap`** (contenu curateur). Les classes **`.err`**, **`.warn`**, **`#authMsg`**, **`.danger-zone`** (et contenu) sont **exclues** des ajustements de taille de police imposés par la densité, et la couleur des erreurs est préservée en mode contraste renforcé, afin de ne pas réduire la lisibilité des messages bloquants ou des zones sensibles.
