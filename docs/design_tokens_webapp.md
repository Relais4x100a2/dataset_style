# Design tokens — slice webapp (issue-007 / issue-022)

## Feuille de style

Les couleurs, espacements et **bandeaux sémantiques** du service `webapp` sont définis dans :

- `src/webapp/static/design_tokens.css` (servi sous `/static/design_tokens.css`)

## Bandeaux (`ds-banner`)

Les retours utilisateur utilisent les classes BEM suivantes :

| Variant (suffixe `--*`) | Usage typique |
|-------------------------|---------------|
| `ds-banner--success` | Succès métier (ex. invitation envoyée en mode SMTP). |
| `ds-banner--warning` | Succès avec vigilance (ex. `MAIL_MODE=dev`, validation sans envoi réel). |
| `ds-banner--danger` | Erreur bloquante ou échec transport (ex. enveloppe API `MAIL_DELIVERY_FAILED`). |
| `ds-banner--info` | Information neutre (ex. anti-IDOR `NOT_FOUND_GENERIC`). |

Le conteneur empilable est `ds-banner-stack` (voir `src/webapp/index_template.py`).

### Bannière migration (`ds-migration-banner`, issue-021 / #184)

Sous-classes pour le texte secondaire et la rangée de liens (sans styles inline) :

- `ds-migration-banner__calendar` — note calendrier sous le message principal.
- `ds-migration-banner__links` — paragraphe des liens d’aide / support.

## Mapping erreurs API → variant

Le script injecte `const API_ERROR_BANNER_VARIANT = { … }` depuis `src/webapp/ui_semantics.py` : chaque `error.code` stable (contrat `docs/api_error_contract.md`) est associé à un des quatre variants ci-dessus. Le frontal **ne** déduit **pas** la sémantique du seul statut HTTP.

## Super-admin (issue-007)

- Sous-onglets : classes `sa-subtabs` / `sa-subtab` (état actif : `sa-subtab active`).
- Périmètre : encadré `sa-scope-lede` pour distinguer visuellement l’administration plateforme du parcours curateur.
- Succès invitation : le JSON `POST /api/super-admin/invite` expose `bannerTone` (`ok` \| `warn`) pour aligner le bandeau sans exposer le jeton ; `inviteResult` (`new_invitation` \| `existing_account_reset`) sert au support et aux tests sans modifier le message succès Streamlit.
