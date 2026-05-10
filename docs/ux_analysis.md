# Analyse UX & Ergonomie — Dataset Style Studio

> **Périmètre** : analyse consolidée — hypothèses produit initiales, audit ligne-par-ligne de `main.py` et `src/ui_components.py`, audit de `src/nlp_engine.py` et `src/export_utils.py`.  
> **Date** : 2026-05-10  
> **Stack** : Streamlit · PostgreSQL · spaCy · SuperTokens · CapRover/Docker  

---

## Table des matières

1. [Proposition de valeur et modèle mental](#1-proposition-de-valeur-et-modèle-mental)
2. [Le moteur stylométrique : différenciateur central — et actuellement invisible](#2-le-moteur-stylométrique--différenciateur-central--et-actuellement-invisible)
3. [Bugs fonctionnels masqués en problèmes UX](#3-bugs-fonctionnels-masqués-en-problèmes-ux)
4. [Hypothèses produit à risque](#4-hypothèses-produit-à-risque)
5. [Incohérences détectées](#5-incohérences-détectées)
6. [Positionnement vs alternatives (Label Studio, Argilla)](#6-positionnement-vs-alternatives-label-studio-argilla)
7. [Backlog priorisé Must / Should / Nice-to-have](#7-backlog-priorisé-must--should--nice-to-have)
8. [Risques majeurs et mitigations](#8-risques-majeurs-et-mitigations)

---

## 1. Proposition de valeur et modèle mental

Dataset Style Studio n'est **pas un outil d'annotation généraliste**. C'est un outil de **curation de datasets de style textuel avec contrôle de cohérence stylométrique en continu**. La distinction est fondamentale pour toutes les décisions UX.

L'objectif produit : produire un dataset fine-tunable dans lequel chaque paire (brouillon, texte réécrit) respecte le même registre stylistique — de sorte que le modèle fine-tuné apprenne un style cohérent, pas une collection de styles hétérogènes.

Pour y parvenir, l'outil embarque un moteur spaCy complet (`src/nlp_engine.py`) qui calcule, pour chaque entrée sauvegardée, une signature stylométrique sur 7 axes et la compare à la moyenne courante du dataset. Cette comparaison produit un **score de cohérence (0-100)** et des **conseils d'écriture priorisés**, calculés en temps réel à chaque sauvegarde. C'est le différenciateur qui rend les alternatives génériques (Label Studio, Argilla, notebooks) non pertinentes pour cet usage.

Le workflow métier réel est donc une **boucle**, pas une séquence linéaire :

```
Créer projet → Configurer dimensions
      ↓
Saisir une entrée (brouillon → réécriture assistée LLM)
      ↓
Sauvegarder → [moteur spaCy] → Score de cohérence + conseils
      ↓
Score ≥ seuil ? → Valider    Score < seuil ? → Réviser l'output
      ↓
Tableau de bord : santé stylométrique du corpus
      ↓
Export JSONL avec indicateurs stylométriques embarqués
```

**Le problème central** : cette boucle est implémentée côté calcul, mais entièrement silencieuse côté UI. L'utilisateur ne voit jamais le score de cohérence, les conseils d'écriture, ni l'état stylométrique du corpus. L'outil se comporte comme un outil d'annotation classique alors qu'il est beaucoup plus.

---

## 2. Le moteur stylométrique : différenciateur central — et actuellement invisible

### 2.1 Ce qui est calculé et stocké

À chaque sauvegarde d'entrée, `compute_row_cache` (`nlp_engine.py` l. 424-466) déclenche une analyse spaCy complète. Les résultats sont stockés dans 12 colonnes de cache sur la table `entries` :

| Colonne | Contenu | Usage actuel |
|---|---|---|
| `_coherence_score` | Score 0-100 de cohérence vs moyenne dataset | Export JSONL uniquement |
| `_signature_json` | Signature 7 axes (JSON) | Export + `signature_variance()` |
| `_ttr` | Type-Token Ratio (richesse lexicale) | Export JSONL uniquement |
| `_ratio` | Ratio expansion input→output | Export JSONL uniquement |
| `_long_phrases` | Longueur moyenne des phrases | Export JSONL uniquement |
| `_syntax_contrast` | Distance syntaxique input vs output (0-1) | Non exposé nulle part |
| `_lexical_density` | Part de mots à contenu fort | Non exposé nulle part |
| `_weak_verb_ratio` | Ratio verbes faibles (être, avoir, faire…) | Non exposé nulle part |
| `_nb_sentences` | Nombre de phrases | Non exposé nulle part |
| `_punct_exp` | Ponctuation expressive (tirets, ellipses, deux-points) | Non exposé nulle part |
| `_stop_ratio_out` | Ratio de mots outils | Non exposé nulle part |
| `_trigrams_json` | Trigrammes POS (empreinte syntaxique) | Non exposé nulle part |

**Fonctions calculables mais non appelées depuis l'UI** :
- `prioritized_actions()` (`nlp_engine.py` l. 275-325) : génère jusqu'à 3 conseils d'écriture concrets ("ta fiche s'écarte de la moyenne sur `Verbes d'action`…")
- `signature_variance()` (`nlp_engine.py` l. 556-590) : écart-type par axe stylistique sur l'ensemble du dataset — identifie les axes hétérogènes
- `coherence_level()` (`nlp_engine.py` l. 248-256) : traduit le score numérique en label qualitatif avec tonalité (Excellent / Bon / À surveiller / Critique)
- `palier_details()` (`nlp_engine.py` l. 223-228) : interprétation textuelle de chaque indicateur (TTR "Bas" = "Vocabulaire répétitif")

### 2.2 Ce qui est visible dans l'UI

Le tableau de bord (`render_tab_dashboard`, `ui_components.py` l. 1031-1054) affiche :

```python
c1.metric("Total", len(df))
c2.metric("Validées", int((df["statut"] == STATUT_VALIDE).sum()))
c3.metric("Types", int(df["type"].nunique()))
st.dataframe(df[["id", "date", "type", "structure", "ton", "format", "public", "statut"]])
```

Soit trois compteurs et un tableau sans aucune colonne stylométrique. **Zéro indicateur de cohérence exposé.**

L'onglet d'édition ne montre pas le score de cohérence de l'entrée courante. L'onglet d'ajout ne donne aucun feedback stylométrique après sauvegarde. L'export ne fournit pas de récapitulatif de santé avant téléchargement.

### 2.3 La conséquence métier

Un utilisateur qui cure 50 entrées peut très bien valider un ensemble stylistiquement hétérogène sans jamais s'en rendre compte. Les indicateurs sont calculés, stockés, et injectés dans les exports JSONL — mais aucune alerte ne remonte pendant la curation. La valeur principale de l'outil est opaque à ceux qui l'utilisent.

De plus, la colonne `_syntax_contrast` (distance syntaxique input→output) est particulièrement critique pour la qualité du fine-tuning : une entrée avec un score proche de 0 signifie que l'output ne transforme presque pas l'input, et que la paire est inutile pour l'apprentissage. Ces entrées ne sont jamais signalées.

---

## 3. Bugs fonctionnels masqués en problèmes UX

Ces points ne sont pas des hypothèses ergonomiques : ce sont des dysfonctionnements vérifiés dans le code.

### 3.1 Génération LLM inopérante dans l'onglet "Nouvelle entrée" (corrigé)

**Localisation** : `src/ui_components.py` — `render_tab_ajout`, helpers `new_entry_session_keys` / `ensure_new_entry_widget_keys_initialized`.

**Historique (bug)** : un `st.form` regroupait `text_area` et trois `form_submit_button`. La génération écrivait dans `session_state["new_generated_output"]` / `["new_generated_input"]` sans lier ces clés aux widgets, donc le texte LLM n'apparaissait pas après rerun.

**Comportement actuel** : plus de `st.form` sur cet onglet ; brouillon et texte généré utilisent des `text_area` avec `key=` stable par projet (`new_entry_{project_id}_input` / `_output`). Les boutons « Générer » mettent à jour les mêmes clés ; « Enregistrer » relit `session_state` au moment du save (alignement affichage / persistance). Les anciennes clés `new_generated_*` sont ignorées puis supprimées à l'entrée de l'onglet.

---

### 3.2 Correction orthographique sans injection dans le formulaire

**Localisation** : `src/ui_components.py` l. 1000-1011 (`render_tab_edition`)

```python
fix = col1.form_submit_button("Corriger output", disabled=disabled)
...
if fix:
    corrected = corriger_texte_fr(...)
    st.info(corrected[:1500] + ("..." if len(corrected) > 1500 else ""))
```

La correction est affichée dans un `st.info` sous le form, mais le champ "Texte généré" n'est pas mis à jour. L'utilisateur doit copier-coller manuellement.

**Correction** : injecter `corrected` dans `st.session_state[f"edit_output_{row['id']}"]` et câbler `value=` du text_area dessus.

---

### 3.3 Incohérence silencieuse CSV vs JSONL à l'export

**Localisation** : `src/ui_components.py` l. 763-782 (`render_tab_settings_export`)

```python
csv        = df[df["statut"] == STATUT_VALIDE].to_csv(...)  # filtre "validé" uniquement
jsonl_data = convert_to_jsonl(df, export_format, ...)        # tout le DataFrame
```

Les deux boutons sont côte à côte sans libellé expliquant la différence de périmètre. L'utilisateur obtient deux fichiers de taille différente sans comprendre pourquoi.

**Correction** : aligner sur le même filtre (préférence : `st.radio("Périmètre", ["Validées seulement", "Tout le dataset"])` appliqué aux deux formats).

---

### 3.4 Feedback stylométrique absent après sauvegarde

**Localisation** : `src/ui_components.py` — `render_tab_edition` et `render_tab_ajout`

Après chaque sauvegarde réussie, le code appelle `update_project_entries` puis `st.rerun()`. Les colonnes de cache `_coherence_score`, `_ttr`, `_syntax_contrast` sont bien calculées et persistées — mais rien n'est affiché à l'utilisateur.

**Conséquence** : la boucle de qualité stylométrique (sauvegarder → voir le score → décider de valider ou réviser) n'existe pas dans l'UI. C'est le dysfonctionnement le plus impactant sur la proposition de valeur de l'outil.

**Correction** : après `update_project_entries`, relire la ligne sauvegardée et afficher le score de cohérence, le TTR et une ligne de conseil issu de `prioritized_actions()`.

---

## 4. Hypothèses produit à risque

### 4.1 "Un seul écran à onglets suffit pour tout le parcours curation"

**Ce que le code confirme** : les 6 onglets (`main.py` l. 62-72) sont au même niveau hiérarchique sans fil conducteur. L'onglet "Nouvelle entrée" est en premier, mais l'utilisateur doit d'abord être passé par les onglets 4 (Projets) et 5 (Réglages & Export) pour que l'outil soit utilisable.

L'ordre actuel des onglets contredit le workflow réel :

| Position | Onglet actuel | Étape workflow réelle |
|---|---|---|
| 1 | Nouvelle entrée | **3e** étape (saisie) |
| 2 | Gestion & édition | **4e** étape (révision) |
| 3 | Tableau de bord | **5e** étape (monitoring cohérence) |
| 4 | Projets | **1re** étape |
| 5 | Réglages & Export | **2e** et **6e** étape |
| 6 | Mon compte | hors workflow principal |

Un utilisateur novice ouvre l'onglet 1 "Nouvelle entrée" avec des dimensions par défaut non configurées, sans avoir compris le modèle projet, et sans aucune introduction au concept de cohérence stylométrique qui est au cœur de l'outil.

---

### 4.2 "Streamlit = ergonomie acceptable pour la saisie intensive"

**Ce que le code confirme** : dans `render_tab_edition` (l. 922-924), la navigation entre entrées est exclusivement via un selectbox non filtrable :

```python
options = [f"{row['id']} · {row['type']} · {row['statut']}" for _, row in df.iterrows()]
idx = st.selectbox("Entrée", list(range(len(options))), format_func=lambda i: options[i])
```

Pour 50+ entrées, il n'y a pas de filtre par statut, pas de filtre par score de cohérence (pour pointer directement les entrées à réviser), pas de navigation clavier, pas d'indicateur de progression. La colonne `_coherence_score` existe en base mais n'est pas utilisée comme critère de tri ou de filtre.

---

### 4.3 "Bloquer l'app sans projet est une bonne première expérience"

**Ce que le code confirme** (`main.py` l. 54-56) :

```python
if not project_id:
    st.info("Crée un projet pour commencer.")
    st.stop()
```

Le message suppose que l'utilisateur a vu la sidebar ouverte et sait où cliquer. Sur mobile ou écran étroit, la sidebar est repliée — le message bloquant arrive sans action corrective visible. Si `DATABASE_URL` n'est pas configurée (`main.py` l. 38-40), le message d'erreur technique `Variable DATABASE_URL requise.` est visible de tout utilisateur, y compris les collaborateurs non techniques.

---

### 4.4 "L'authentification invitation-only simplifie l'UX"

**Ce que le code confirme** : le nouvel invité arrive sur l'interface sans aucun contexte — pas de message de bienvenue, pas de description de la proposition de valeur stylométrique, pas de checklist de démarrage. Il atterrit directement sur l'écran de blocage "Crée un projet pour commencer." sans comprendre ni le modèle projet, ni l'objectif de l'outil.

---

### 4.5 "La charge de données est légère"

**Ce que le code confirme** : `main.py` l. 58 exécute `load_project_entries(engine, project_id, user.user_id)` à chaque rerun Streamlit, sans `@st.cache_data`. Chaque interaction utilisateur déclenche une requête SQL qui rapatrie toutes les colonnes, dont les 12 colonnes de cache (`_signature_json`, `_trigrams_json`…) qui peuvent être volumineuses. Pour 300+ entrées, la latence perçue peut devenir significative.

---

## 5. Incohérences détectées

### 5.1 Le tableau de bord ne reflète pas la mission de l'outil

`render_tab_dashboard` (l. 1031-1054) affiche trois compteurs (Total, Validées, Types) et un tableau sans colonnes stylométriques. La mission déclarée de l'outil est de produire un dataset stylistiquement cohérent, mais le dashboard ne permet pas de savoir si le dataset en cours est cohérent ou non. C'est l'incohérence la plus structurante entre la promesse produit et l'interface.

---

### 5.2 Double lieu de gestion projet

La sidebar (`render_sidebar`, l. 423-461) contient le sélecteur de projet et le formulaire de création de premier projet. L'onglet "Projets" (`render_tab_projects`, l. 464-490) contient aussi un formulaire de création. Ces deux surfaces se chevauchent. Un utilisateur avec 0 projet ne voit jamais l'onglet "Projets" (l'app s'arrête avant sur `st.stop()`).

**Clarification suggérée** : sidebar = contexte uniquement (sélecteur + rôle), onglet Projets = toutes les actions (créer, supprimer, membres).

---

### 5.3 L'onglet Super Admin mélange deux profils d'usage

`render_tab_super_admin` (l. 544-743) concentre ~200 lignes avec :
- **profil opérationnel** : inviter un utilisateur, gérer les comptes (suppression, detach memberships)
- **profil technique** : monitoring saga comptes (états pending/failed/quarantined), DLQ replay

Un super admin qui veut juste inviter un collaborateur se retrouve face à des tableaux de monitoring technique qui ne le concernent pas.

---

### 5.4 Libellés mélangent le technique et le métier

- "LLM base URL", "LLM model", "LLM API key", "LanguageTool base URL" sont exposés directement dans l'onglet Réglages (`_render_project_settings_form`, l. 160-222).
- "Detach memberships", "DLQ", "Quarantined", "Replay opération", "saga comptes" sont du vocabulaire d'infrastructure visible dans l'UI Super Admin.
- `operation_id`, `target_user_id` apparaissent directement dans les tableaux de l'interface Super Admin.

---

### 5.5 Absence de feedback sur les actions destructives réussies avant rerun

Les suppressions (projet, compte) se terminent par `st.success(...)` + `st.rerun()` immédiat. Le `rerun` peut effacer le message avant que l'utilisateur l'ait lu. Le pattern correct est de stocker le message dans `session_state` et de l'afficher au render suivant.

---

## 6. Positionnement vs alternatives (Label Studio, Argilla)

### Pourquoi Label Studio ne couvre pas le cas d'usage central

Label Studio est un outil d'annotation généraliste performant sur le workflow de révision (navigation clavier, filtres, multi-utilisateur). Il couvre correctement 70-80% des fonctionnalités secondaires de Dataset Style Studio.

Mais il est **incompatible avec le différenciateur central** :

- Le score de cohérence stylométrique est calculé **relativement à la moyenne courante du dataset validé**. Cette invariant nécessite un moteur spaCy couplé à chaque sauvegarde individuelle, avec accès au corpus entier.
- `signature_variance()` mesure l'hétérogénéité stylométrique de l'ensemble du corpus — c'est une propriété émergente du dataset, pas d'une entrée individuelle.
- `prioritized_actions()` génère des conseils d'écriture personnalisés en comparant la signature de la fiche courante aux axes où le dataset dérive.
- `_syntax_contrast` détecte les paires input→output trop proches, inutiles pour le fine-tuning — un signal qualité qui n'existe dans aucun outil d'annotation standard.

Reproduire cette boucle dans Label Studio via un ML Backend est techniquement possible, mais revient à réécrire `nlp_engine.py` en tant que service externe, sans l'intégration native UI (affichage inline du score, conseils dans le formulaire, dashboard de variance). Le coût de développement serait supérieur à développer Dataset Style Studio correctement.

### Tableau comparatif

| Fonctionnalité | Label Studio | Dataset Style Studio (actuel) | Dataset Style Studio (cible) |
|---|---|---|---|
| Navigation entre entrées (filtres, clavier) | Natif, excellent | Faible (selectbox) | Should S3 + N1 |
| Multi-projet / rôles | Natif | Natif | — |
| Score de cohérence stylométrique | Impossible nativement | Calculé, invisible | Must M5 |
| Conseil d'écriture inline | Impossible | Calculé, invisible | Must M5 |
| Dashboard variance stylométrique | Absent | Calculé, invisible | Must M6 |
| Filtre "entrées à risque" (score bas) | Absent | Données disponibles | Should S3 |
| Détection paires triviales (syntax_contrast) | Absent | Calculé, invisible | Should S8 |
| LLM-assisted generation inline | ML Backend requis | Implémenté (bug M1) | Must M1 |
| Export lfm2 / baguettotron / mistral | Script post-export | Natif | — |
| Indicateurs stylométriques dans JSONL | Absent | Natif | — |

**Verdict** : Label Studio est pertinent uniquement si la cohérence stylométrique est retirée du scope produit. Ce choix viderait l'outil de son différenciateur.

---

## 7. Backlog priorisé Must / Should / Nice-to-have

### Must — différenciateur et correctifs bloquants

| # | Action | Localisation | Impact |
|---|--------|--------------|--------|
| M1 | Corriger la génération LLM : sortir les `text_area` du `st.form`, câbler sur `session_state` | `ui_components.py` l. 814-898 | Fonctionnalité cassée → utilisable |
| M2 | Corriger la correction orthographique : injecter le résultat dans le form via `session_state` | `ui_components.py` l. 1000-1011 | Fonctionnalité dégradée → utilisable |
| M3 | Aligner périmètre CSV / JSONL à l'export | `ui_components.py` l. 763-782 | Incohérence silencieuse → comportement prévisible |
| M4 | Remplacer `st.error("Variable DATABASE_URL requise.")` par un message non technique | `main.py` l. 38-40 | Erreur technique visible par tous → message adapté |
| **M5** | **Afficher le score de cohérence après chaque sauvegarde** : relire `_coherence_score`, `_ttr`, `_syntax_contrast` depuis la ligne persistée, appeler `prioritized_actions()`, afficher via `st.metric` + `st.info` | `ui_components.py` l. 870-898, 1011-1028 | **Le différenciateur devient visible pendant la curation** |
| **M6** | **Tableau de bord de santé stylométrique** : remplacer les 3 compteurs actuels par distribution des scores de cohérence, résultat de `signature_variance()` par axe, top 5 entrées outliers (`_coherence_score` bas), moyenne `_syntax_contrast` | `ui_components.py` l. 1031-1054 | **La qualité du corpus devient mesurable et actionnable** |

### Should — améliorations workflow

| # | Action | Localisation | Impact |
|---|--------|--------------|--------|
| S1 | Réordonner les onglets dans l'ordre du workflow réel | `main.py` l. 62-72 | Orientation naturelle dès la première session |
| S2 | Remplacer `st.stop()` "Crée un projet" par un écran d'accueil guidé en 3 étapes | `main.py` l. 54-56 | Onboarding nouveau → abandon réduit |
| S3 | Ajouter filtre par statut **et par score de cohérence** dans l'onglet édition | `ui_components.py` l. 922-924 | Pointer directement les entrées à réviser |
| S4 | Ajouter `@st.cache_data(ttl=30)` sur `load_project_entries` avec invalidation après écriture | `main.py` l. 58 | Latence perçue réduite |
| S5 | Clarifier la responsabilité sidebar (contexte seul) vs onglet Projets (actions seules) | `render_sidebar` + `render_tab_projects` | Double-lieu éliminé |
| S6 | Séparer Super Admin : "Gestion comptes" visible, "Monitoring technique" en `st.expander` fermé par défaut | `ui_components.py` l. 544-743 | Charge cognitive réduite |
| S7 | Persister les messages de succès via `session_state` avant `st.rerun` | `ui_components.py` l. 415-418, 535-539 | Message de confirmation lisible |
| S8 | Signaler les entrées avec `_syntax_contrast < 0.2` (paires trop proches, inutiles pour le fine-tuning) dans le dashboard et dans l'édition | `nlp_engine.py` `syntax_contrast_score` | Qualité dataset → fine-tuning plus efficace |

### Nice-to-have — confort et polish

| # | Action | Impact |
|---|--------|--------|
| N1 | Navigation précédent/suivant dans l'édition (boutons) | Curation séquentielle plus rapide |
| N2 | Indicateur de progression ("Entrée 12/47 · 8 à réviser · 3 outliers") | Sentiment de contrôle |
| N3 | Libellés techniques traduits en français métier ("Modèle d'IA" au lieu de "LLM model") | Accessibilité non-technique |
| N4 | Email d'invitation avec description en une phrase de la proposition de valeur stylométrique | Onboarding email → contexte immédiat |
| N5 | Rapport de cohérence avant export : "47 entrées validées · cohérence moyenne 81/100 · 3 outliers à vérifier" | Confiance avant export |
| N6 | Tests utilisateur courts (3-5 sessions) sur le parcours : invitation → projet → première entrée → lecture score cohérence → validation | Données terrain pour prioriser les itérations suivantes |

---

## 8. Risques majeurs et mitigations

### Risque 1 — Valeur différenciante invisible = outil perçu comme générique

**Cause** : le moteur stylométrique calcule des scores sophistiqués qui ne sont jamais montrés. Les utilisateurs perçoivent l'outil comme un formulaire de saisie avec dimensions — exactement ce que Label Studio ferait mieux. Sans M5 et M6, le différenciateur n'existe pas pour les utilisateurs.

**Mitigation** : M5 (score inline post-save) et M6 (dashboard de santé) sont les deux items de backlog les plus structurants. Tout le reste est secondaire par rapport à rendre visible ce qui est déjà calculé.

---

### Risque 2 — Fonctionnalité LLM perçue comme instable

**Cause** : le bug M1 (génération silencieusement inopérante) est probablement connu des utilisateurs actuels sans avoir été identifié comme un bug. Ils ont peut-être développé des contournements ou abandonné la fonctionnalité.

**Mitigation** : corriger M1/M2 et communiquer explicitement le correctif. Critère de succès : l'utilisateur clique "Générer texte" et le résultat s'affiche dans la zone de saisie sans action manuelle.

---

### Risque 3 — Fatigue cognitive / dispersion des fonctions

**Cause** : 6 onglets de niveau équivalent sans fil conducteur, ordre contre-intuitif, double surface de gestion projet.

**Mitigation** : S1 (réordonnancement) + S5 (clarification sidebar/onglet). Tester avec 2-3 utilisateurs sur la tâche "créer un projet, configurer une dimension, ajouter une entrée, lire le score de cohérence" — mesurer le temps jusqu'à la première entrée validée avec compréhension du score.

---

### Risque 4 — Perte de contexte lors des re-runs Streamlit

**Cause** : chaque action déclenche un rerun. Sans `session_state` comme source de vérité pour les champs de saisie longue, les utilisateurs perdent leur texte en cours.

**Mitigation** : usage systématique de `session_state` pour les champs `input`/`output` (couvert par la correction M1). Regrouper les opérations atomiques dans `st.form`, sortir les opérations assistées (LLM, spaCy) du form.

---

### Risque 5 — Onboarding faible pour les nouveaux invités

**Cause** : l'email d'invitation ne décrit pas le produit. L'interface ne propose aucun écran de bienvenue ni explication du concept de cohérence stylométrique. Le premier blocage (sans projet) n'est pas guidant.

**Mitigation** : S2 (écran guidé) + N4 (email avec premier clic explicite et mention de la proposition de valeur). Critère de succès : un invité peut créer un projet, enregistrer une première entrée et lire son score de cohérence sans assistance en moins de 5 minutes.

---

### Risque 6 — Dataset hétérogène exporté sans signal d'alerte

**Cause** : en l'état, un utilisateur peut valider 50 entrées stylistiquement disparates et exporter un JSONL sans jamais recevoir d'alerte sur la qualité. Les indicateurs `_coherence_score` et `signature_variance()` sont calculés mais non communiqués. Le modèle fine-tuné sur ce dataset apprendra des styles mixtes.

**Mitigation** : M6 (dashboard de santé) + N5 (rapport avant export). Définir un seuil d'alerte (ex. cohérence moyenne < 65 = `st.warning` avant les boutons de téléchargement). Ce risque est le plus directement lié à la qualité des modèles produits.

---

### Risque 7 — Attentes "outil data moderne" non satisfaites

**Cause** : des utilisateurs habitués à Label Studio, Argilla, ou Prodigy s'attendront à de la collaboration temps réel, des commentaires par entrée, un historique des modifications, et des raccourcis clavier. Dataset Style Studio ne couvre pas ces usages.

**Mitigation** : documenter et communiquer le positionnement explicite de l'outil autour de la cohérence stylométrique, pas de l'annotation généraliste. Ce positionnement est la réponse à "pourquoi pas Label Studio" — il doit être visible dans l'interface elle-même (titre, description, dashboard).

---

*Document produit à partir de l'audit de `main.py`, `src/ui_components.py` (1055 lignes), `src/nlp_engine.py` (591 lignes) et `src/export_utils.py` (166 lignes). Les références de ligne correspondent à l'état du code au 2026-05-10.*
