# Contrat d’erreurs API (issue-005)

Objectif : réponses **JSON homogènes**, **codes stables** consommables par le futur frontal (toasts / issue-022), messages **en français**, sans s’appuyer sur le texte SuperTokens ou SQL. Référence code : `src/api_errors.py`.

## Enveloppe JSON (alignement issue-006)

Toute erreur renvoyée au client suit :

```json
{
  "error": {
    "code": "NOT_FOUND_GENERIC",
    "title": "…",
    "message": "…",
    "suggested_action": "…",
    "detail": null
  }
}
```

- **`code`** : identifiant stable (snake_case SCREAMING).
- **`title`**, **`message`**, **`suggested_action`** : chaînes FR pour l’UI.
- **`detail`** : `null` en **production** ; renseigné uniquement en **développement** (même règle que l’UI Streamlit via `is_development_ui()` dans `src/db_startup.py`).

## Règle prod vs dev

- **Prod** : pas de stack trace ni de corps d’erreur provider/SQL dans la réponse ; journalisation structurée côté serveur (`api_error_code`, `ctx_*`).
- **Dev** : `detail` peut contenir le type d’exception et un extrait de message, avec masquage basique des secrets (voir `technical_detail_text` dans `src/api_errors.py`).

## Politique anti-IDOR (ressources tenantées)

Pour tout accès à une ressource **hors périmètre** (projet inexistant pour l’utilisateur, rôle insuffisant, etc.), l’API expose **toujours** le même couple :

- **HTTP** : `404`
- **`code`** : `NOT_FOUND_GENERIC`

Liste et détail doivent appliquer **strictement** la même réponse (pas de `403` + message différent qui révélerait l’existence d’un projet). Côté données, lever `TenantResourceOpaqueDenial` (`src/api_errors.py`) depuis `require_role` / garde-fous équivalents.

## Taxonomie (codes stables)

| Code | HTTP | Titre (FR) | Message (FR) | Action suggérée (FR) |
|------|------|------------|----------------|----------------------|
| `AUTH_SESSION_EXPIRED` | 401 | Session expirée | Votre session a expiré ou n'est plus valide. | Déconnectez-vous puis reconnectez-vous. |
| `DB_UNAVAILABLE` | 503 | Service de données indisponible | *(corps aligné sur `user_facing_summary` / démarrage DB)* | Réessayez dans quelques minutes ; si le problème persiste, contactez l'administrateur. |
| `FORBIDDEN` | 403 | Accès refusé | Vous n'avez pas les droits suffisants pour cette opération. | Si vous pensez qu'il s'agit d'une erreur, contactez un administrateur. |
| `NOT_FOUND_GENERIC` | 404 | Ressource introuvable | Cette ressource n'existe pas, n'est plus disponible, ou vous n'y avez pas accès. | Vérifiez votre sélection ou l'URL ; reconnectez-vous si besoin. |
| `INTERNAL_ERROR` | 500 | Erreur interne | Une erreur technique s'est produite. | Réessayez plus tard. Si le problème persiste, contactez l'administrateur. |

Les entrées du tableau correspondent aux constantes et textes dans `src/api_errors.py` (source de vérité pour le mapping).

## Intégration FastAPI (futur BFF)

- Importer `error_envelope_for_client(exc, include_technical_detail=None)` et renvoyer le dict comme corps JSON avec le statut `resolve_exception_for_api(...).http_status`.
- Ne pas exposer `HTTPException.detail` texte brut provider : passer toujours par ce module.

## UI Streamlit (préparation issue-022)

- `src/flash_messages.py` accepte un champ optionnel **`code`** sur les flashes post-rerun ; le rendu affiche une légende `code: …` pour corrélation support / futur mapping toasts.

## Slice webapp (issue-022 livré)

- Fichier CSS : `src/webapp/static/design_tokens.css` ; mapping Python `error.code` → variant : `src/webapp/ui_semantics.py`.
- Guide produit / contraste / XSS : `docs/design_tokens_webapp.md`.
