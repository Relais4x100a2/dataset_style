# Design tokens — slice webapp (issue-007 / issue-011 / issue-022)

Ce document recense les **tokens et classes transverses** de la coquille FastAPI. Les couleurs, espacements et rayons des composants partagés **ne doivent pas** être dupliqués en hex arbitraires dans les gabarits ou le HTML généré par le JavaScript : utiliser `var(--ds-…)` ou les classes `.ds-*`.

## Feuille de style

| Fichier | Rôle |
|---------|------|
| `src/webapp/static/design_tokens.css` | Variables `:root`, composants (boutons, bandeaux, tables, skeleton), médias `prefers-contrast`, `forced-colors`, `prefers-reduced-motion` |
| `src/webapp/index_template.py` | HTML shell ; lien `<link rel="stylesheet" href="/static/design_tokens.css" />` ; pas de bloc `<style>` inline |
| Montage FastAPI | `StaticFiles` sur `/static` dans `src/webapp/app.py` |

Servi sous `/static/design_tokens.css`.

## Vision UI — tokens transverses (issue-011 / #185)

| Jeton / concept | Statut | Notes |
|-----------------|--------|-------|
| `--ds-color-action-fill` (+ hover/active) | **Ajouté** | Fond bouton primaire ; texte `--ds-color-on-action-fill` ; contraste visé **WCAG AA** |
| Boutons **primary / secondary / danger** (`.ds-btn--*`) | **Ajouté** | États `:hover`, `:active`, `:focus-visible`, `:disabled` ; overrides `prefers-contrast: more` et `forced-colors` |
| `--ds-space-*` (1 → 6) | **Ajouté** | Échelle fixe pour marges/padding |
| `--ds-radius-sm` | **Ajouté** | Rayons boutons et composants |
| `--ds-color-row-hover`, `--ds-color-row-border` | **Ajouté** | Listes / tableaux (`tr.entries-row-openable`) |
| `--ds-chip-*` + classe `.ds-chip` | **Ajouté** | Badges légers |
| Skeleton (`.ds-skeleton-line`, `--ds-skeleton-*`) | **Ajouté** | Barre animée ; sous `prefers-reduced-motion` : animation désactivée |
| **Densité / confort d'affichage** | **Issue-023** | Attributs `data-ds-density` / `data-ds-reading` sur `<html>` ; voir `docs/ui_display_preferences.md` |

## Bandeaux (`ds-banner`, issue-022)

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
- `ds-migration-banner__links` — paragraphe des liens d'aide / support.

## Mapping erreurs API → variant

Le script injecte `const API_ERROR_BANNER_VARIANT = { … }` depuis `src/webapp/ui_semantics.py` : chaque `error.code` stable (contrat `docs/api_error_contract.md`) est associé à un des quatre variants ci-dessus. Le frontal **ne** déduit **pas** la sémantique du seul statut HTTP.

## Super-admin (issue-007)

- Sous-onglets : classes `sa-subtabs` / `sa-subtab` (état actif : `sa-subtab active`).
- Périmètre : encadré `sa-scope-lede` pour distinguer visuellement l'administration plateforme du parcours curateur.
- Succès invitation : le JSON `POST /api/super-admin/invite` expose `bannerTone` (`ok` \| `warn`) pour aligner le bandeau sans exposer le jeton ; `inviteResult` (`new_invitation` \| `existing_account_reset`) sert au support et aux tests sans modifier le message succès Streamlit.

## Accessibilité

- **`prefers-contrast: more`** : bordures renforcées, primaire/danger assombris.
- **`forced-colors: active`** : boutons mappés sur `ButtonText` / `Mark` pour rester utilisables.
- **`prefers-reduced-motion: reduce`** : transitions des boutons supprimées ; shimmer skeleton désactivé.

Toute nouvelle surface HTML (y compris chaînes construites en JS) doit s'appuyer sur ce fichier ou étendre **ici** la liste des tokens avant d'introduire des couleurs ad hoc.
