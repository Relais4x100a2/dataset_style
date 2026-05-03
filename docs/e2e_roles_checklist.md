# Checklist E2E Rôles

1. Créer un compte `A` via SuperTokens.
2. Connexion `A`, création des projets `P1` et `P2`.
3. Vérifier que `A` peut basculer entre `P1` et `P2` depuis la sidebar.
4. Vérifier que les entrées affichées sont bien isolées par projet.
5. Vérifier que les réglages LLM/LanguageTool sont propres au projet courant.
6. Vérifier suppression projet avec double confirmation (checkbox + nom du projet).
7. Vérifier export CSV/JSONL depuis l'onglet `Réglages & Export`.
8. Vérifier login/logout.

## Presets & Legacy (UX v2)

### A. Legacy préservé
1. Créer une entrée avec `format="Chapitre"` en preset `roman`.
2. Basculer sur preset `pro`, puis cliquer `Charger le preset`.
3. Ouvrir l'édition de l'entrée:
   - vérifier l'option `[obsolète] Chapitre` pré-sélectionnée,
   - vérifier l'avertissement "Cette valeur existe dans vos données mais plus dans le preset actif."
4. Sauvegarder sans changer ce champ.
5. Vérifier que la valeur `Chapitre` est toujours présente en base.

### B. Preset actif cohérent
1. Sélectionner preset `contenu`.
2. Cliquer `Réinitialiser`.
3. Recharger la page.
4. Vérifier:
   - `active_preset_key == contenu`,
   - dimensions alignées sur le preset `contenu`.

### C. Feedback génération
1. Dans `Nouvelle entrée`, cliquer `Générer texte` puis `Générer brouillon`.
2. Vérifier l'affichage du spinner "Génération en cours...".
3. Vérifier un retour explicite en succès ou en erreur actionnable.

