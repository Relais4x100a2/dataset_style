# Design tokens et bandeaux sémantiques (issue-022 / #144)

Phase 1 : charte visuelle **légère** pour le slice FastAPI / coquille HTML (`GET /`), sans design system complet.

## Fichiers

| Élément | Emplacement |
|--------|-------------|
| Variables CSS (surfaces, texte, sémantique) | `src/webapp/static/design_tokens.css` |
| Servi sous | `GET /static/design_tokens.css` (montage FastAPI) |
| Mapping `error.code` → variant bandeau | `src/webapp/ui_semantics.py` |
| Injection client (objet JS) | `index_template.INDEX_HTML` (placeholder `__API_ERROR_BANNER_VARIANT_JSON__`) |

Polices : **pile système uniquement** (`system-ui`, `-apple-system`, etc.) — pas de Google Fonts, compatible avec une CSP future stricte.

## Variants bandeau

| Classe | Usage |
|--------|--------|
| `ds-banner--success` | Confirmation (connexion, enregistrement, absence d’alerte qualité, relance saga OK). |
| `ds-banner--warning` | **Qualité dataset** (stylométrie / cohérence), session à renouveler, requête refusée pour droits, validation utilisateur, invitation en mode mail simulé. |
| `ds-banner--danger` | **Erreur technique** ou incident serveur (base indisponible, erreur interne, service tiers critique), code API inconnu. |
| `ds-banner--info` | Information neutre (ressource introuvable côté anti-IDOR, sélection requise, alertes `severity: info` du dashboard). |

### Quand « warning » (qualité) vs « danger » (technique / destructif)

- **Warning** : le système fonctionne mais signal métier ou action utilisateur attendue (seuil stylométrie, scores manquants, session expirée, export trop volumineux, `BAD_REQUEST`, `FORBIDDEN`, `CLIENT`).
- **Danger** : défaillance ou risque fort côté plateforme (`DB_UNAVAILABLE`, `INTERNAL_ERROR`, `CURATOR_LANGUAGETOOL_UNAVAILABLE` lorsque le service est injoignable, code non catalogué → défaut danger).

Les **actions destructives** (ex. zone DLQ saga) restent dans la **danger-zone** dédiée ; le bandeau associé utilise le variant pertinent selon le résultat (succès / erreur API).

## Contraste (WCAG 2.1 AA)

Les combinaisons **texte principal sur fond** des quatre variants visent le rapport **≥ 4,5:1** (texte normal). Les couleurs de lien / focus reprennent des teintes plus soutenues sur le même fond (voir variables `--ds-banner-*-link` et `--ds-color-focus-ring`).

Révision manuelle recommandée si les tokens sont modifiés (outil type WebAIM Contrast Checker).

## Sécurité (XSS)

Les messages API et les textes d’alertes qualité sont rendus en **texte** via `textContent` / nœuds DOM (pas d’`innerHTML` avec chaînes non fiables sur les bandeaux). La bannière migration (`APP_MIGRATION_INFO_BANNER`) reste passée par `html.escape` côté Python (`migration_communication.py`).

## Intégration issue-005

Le client ne déduit **pas** le variant à partir du seul statut HTTP : il lit `error.code` et applique le mapping aligné sur `src/webapp/ui_semantics.py` (identique à l’objet injecté dans la page pour le navigateur).
