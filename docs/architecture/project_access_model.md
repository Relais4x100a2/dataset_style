# Modèle d’accès projet (issue-003)

Décision d’architecture : aligner la narration produit (**un projet a un seul propriétaire**) avec le schéma PostgreSQL et le comportement effectif de l’application, pour éviter une double logique d’accès lors des évolutions (nouveau frontal, BFF, APIs).

## Décision

1. **Propriétaire unique**  
   - Chaque projet a **exactement un** compte propriétaire : `projects.created_by` → `users.id`.  
   - C’est la source de vérité pour la **propriété** (qui peut supprimer le projet, qui est bloqué tant qu’il possède des projets, libellés produit « propriétaire »).

2. **`project_memberships` : collaboration préparée, pas un second propriétaire**  
   - La table `(project_id, user_id, role)` sert à enregistrer des **rôles de collaboration** (`admin`, `collaborator`, `viewer` — voir `PROJECT_ROLES` dans `src/database.py`) **sans** créer un second « propriétaire » du dataset.  
   - Aucune ligne de membership ne remplace ni ne duplique `created_by` pour la notion produit « propriétaire du projet ».

3. **Autorisation runtime aujourd’hui (Streamlit + `src/database.py`)**  
   - `get_role()` ne considère **que** le propriétaire : si `user_id == projects.created_by` et le projet n’est pas archivé, le rôle effectif exposé aux gardes-fous est **`admin`** (propriétaire). Sinon, `None`.  
   - Les signatures `require_role(..., ("admin", "collaborator", "viewer"))` sur chargement / écriture d’entrées préparent une **future** prise en compte des rôles membership ; **tant que `get_role` n’interroge pas `project_memberships`, seul le propriétaire passe les contrôles** pour les données du dataset.  
   - C’est cohérent avec le message produit « pas de partage multi-membres **fonctionnel** actif » : la table existe, les flux admin (invitation de comptes, détachement super-admin) peuvent manipuler des lignes, mais **l’accès effectif aux entrées du projet suit la règle propriétaire ci-dessus**.

4. **Super-admin (`users.is_super_admin`)**  
   - Périmètre **global** : gouvernance des comptes (invitation, liste, saga de déprovisionnement, détachement des memberships, promotion premier admin).  
   - **Pas** un co-propriétaire implicite des projets d’autrui pour les opérations courantes du curateur : les actions sur données projet (`load_project_entries`, `require_admin`, etc.) restent calées sur `get_role` / propriétaire, sauf évolution produit explicite documentée ici et implémentée partout (UI + couche données + futur BFF).

## Feuille de route

| Phase | Contenu |
|--------|---------|
| **Actuelle** | Propriété = `created_by` ; memberships = persistance collaboration + opérations admin ; `get_role` = propriétaire uniquement. |
| **Évolution partage (si produit l’exige)** | Étendre **une seule** primitive d’autorisation (ex. `get_role` ou équivalent BFF) pour lire `project_memberships` et renvoyer le rôle réel ; aligner `list_projects_for_user` (ou équivalent API) pour lister les projets « invités » ; éviter que l’UI et le backend calculent des droits différents. |
| **Dépréciation** | Non retenue à ce stade : la table reste le support naturel du partage ; si le produit abandonnait le partage, on pourrait figer l’usage à « interne / vide » après migration de nettoyage — hors périmètre tant qu’aucune décision produit contraire. |

**Migration SQL** : aucune requise pour cette décision (clarification comportementale + doc uniquement).

## Implications UI / API (checklist)

- **Champs exposés (futur BFF / API)**  
  - Toujours exposer un **`owner_user_id`** (ou équivalent) dérivé de `projects.created_by`.  
  - Exposer `membership_role` **uniquement** si la couche d’auth lit réellement `project_memberships` pour ce projet et cet utilisateur ; sinon documenter « non applicable » pour éviter les clients qui afficheraient un rôle fantôme.

- **Comportement si rôle collaborateur non offert aux curateurs**  
  - Tant que `get_role` est propriétaire-seul : ne pas présenter de parcours « j’accède au projet X en tant que collaborateur » côté liste de projets sans implémenter la liste jointe owner ∪ memberships et la même règle côté serveur.

- **Libellés**  
  - « Propriétaire » = `created_by` ; « collaborateur / viewer » = membership, sous réserve que l’évolution partage soit livrée.

## Revue routes / issue-010

L’application actuelle est **Streamlit** : pas de routes HTTP applicatives distinctes dans ce dépôt. Toute **nouvelle** surface (REST, BFF, etc.) doit :

- réutiliser la même sémantique **un propriétaire par projet** ;
- ne pas introduire d’endpoint qui confondrait « membre avec rôle admin dans `project_memberships` » avec « propriétaire du projet » sans le documenter explicitement.

La revue **issue-010** doit inclure une vérification explicite de ce point.

## Références code

- Schéma et gardes-fous : `src/database.py` (`ensure_schema`, `get_role`, `require_role`, `project_memberships`, `list_projects_for_user`).  
- Vue d’ensemble multi-tenant : `docs/multi_tenant_architecture.md`.
