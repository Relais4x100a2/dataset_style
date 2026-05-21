# Tokens design — webapp curateur

Ce document recense les **tokens et classes transverses** de la coquille FastAPI (`src/webapp/static/design_tokens.css`). Les couleurs, espacements et rayons des composants partagés **ne doivent pas** être dupliqués en hex arbitraires dans les gabarits ou le HTML généré par le JavaScript : utiliser `var(--ds-…)` ou les classes `.ds-*`.

## Fichier canonique

| Fichier | Rôle |
|---------|------|
| `src/webapp/static/design_tokens.css` | Variables `:root`, composants (boutons, tables, skeleton), médias `prefers-contrast`, `forced-colors`, `prefers-reduced-motion` |
| `src/webapp/index_template.py` | HTML shell ; lien `<link rel="stylesheet" href="/static/design_tokens.css" />` ; pas de bloc `<style>` inline |
| Montage FastAPI | `StaticFiles` sur `/static` dans `src/webapp/app.py` |

## Vision UI — état par token

| Jeton / concept | Statut | Notes |
|-----------------|--------|-------|
| `--ds-color-action-fill` (+ hover/active) | **Ajouté** | Fond bouton primaire ; texte `--ds-color-on-action-fill` ; contraste visé **WCAG AA** (texte clair sur fond bleu) |
| `--ds-color-on-action-fill` | **Ajouté** | Texte/icônes sur primaire rempli |
| Boutons **primary / secondary / danger** (`.ds-btn--*`) | **Ajouté** | États `:hover`, `:active`, `:focus-visible`, `:disabled` ; overrides `prefers-contrast: more` et `forced-colors` |
| `--ds-space-*` (0 → 8) | **Ajouté** | Échelle fixe pour marges/padding ; **pas** de tokens de densité/confort (réservés **issue-023**) |
| `--ds-radius-sm/md/pill` | **Ajouté** | Rayons boutons, panneaux, nav |
| `--ds-color-row-hover`, `--ds-color-row-border`, `--ds-color-row-selected-bg` | **Ajouté** | Listes / tableaux (`tr.entries-row-openable`, `table.*`) |
| `--ds-chip-*` (bg, fg, border, radius, padding) + classe `.ds-chip` | **Ajouté** | Badges légers ; prêt pour écrans denses 012/013/017 |
| Skeleton chargement (`.ds-skeleton-line`, variables `--ds-skeleton-*`) | **Ajouté** | Barre animée sous **réduction de mouvement** : animation désactivée, fond uni |
| Couleurs canvas / surface / bordures / texte | **Ajouté** | `--ds-color-canvas`, `--ds-color-surface`, `--ds-color-border-*`, `--ds-color-fg-*` |
| Sémantique erreur / succès / avertissement | **Ajouté** | `.err`, `.ok`, `.warn`, zone `.danger-zone` |
| **Densité / confort d’affichage** (ex. `--ds-density-*`, espacement utilisateur) | **Reporté** | **Issue-023** : préférences utilisateur distinctes des tokens de base |

## Accessibilité

- **Contraste** : les combinaisons primaire (fond `--ds-color-action-fill` + texte blanc) et danger (fond `--ds-color-danger-fill` + blanc) sont choisies pour viser **AA** sur fond clair ; en cas de doute, vérifier avec un calculateur (rapport ≥ 4,5:1 pour texte courant).
- **`prefers-contrast: more`** : bordures renforcées, primaire/danger assombris.
- **`forced-colors: active`** : boutons mappés sur `ButtonText` / `Mark` pour rester utilisables.
- **`prefers-reduced-motion: reduce`** : transitions des boutons supprimées ; shimmer skeleton désactivé.

## Évolution

Toute nouvelle surface HTML (y compris chaînes construites en JS) doit s’appuyer sur ce fichier ou étendre **ici** la liste des tokens avant d’introduire des couleurs ad hoc.

Référence issue : **#185** (issue-011 backlog orchestrateur).
