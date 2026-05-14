# Règle produit — navigation curateur (issue-011)

## Intitulé

**Une seule surface officielle** pour créer, consulter et supprimer des projets : l’onglet **Projets** dans la zone principale. Le menu **☰** (barre latérale) reste centré sur le **contexte** : compte, déconnexion, et choix du **projet courant** lorsque des projets existent déjà.

## Pourquoi

Éviter deux « endroits officiels » concurrents pour la création de projet (sidebar vs onglet), qui fragmentent le modèle mental et compliquent l’onboarding (issues **008**, **028**). Cette règle est le socle UX pour le déploiement multi-tenant CapRover décrit dans le README.

## Où c’est implémenté

| Élément | Emplacement |
|---------|-------------|
| Texte long (légende d’accueil sans projet) | `PRODUCT_RULE_ISSUE_011_CREATION_PATHS_FR` dans `src/empty_project_onboarding.py` |
| Rappel court (astuce) | `SIDEBAR_CONTEXT_HINT_FR` dans le même module |
| Affichage onboarding | `render_no_project_onboarding` dans `src/ui_components.py` |
| Message sans projet dans la sidebar | `render_sidebar` dans `src/ui_components.py` |

## Critères de recette (QA)

1. Utilisateur sans projet : l’accueil dans l’onglet **Projets** affiche la règle (captions) et le formulaire de premier projet dans la zone principale, pas dans la sidebar.
2. Sidebar sans projet : message orientant vers l’onglet **Projets** ; pas de formulaire de création dans la sidebar.
3. Après création du premier projet : le menu **☰** permet de choisir le projet courant ; création / suppression d’autres projets reste dans **Projets**.

## Tests automatisés

`tests/test_empty_project_onboarding.py` — `test_product_rule_issue_011_separates_context_sidebar_and_actions_tab`.
