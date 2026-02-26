# Stylométrie et insights pour le fine-tuning

Recommandations d’expert NLP sur les **statistiques stylométriques pertinentes** à stocker par ligne (Google Sheets) pour les réutiliser comme **insights** lors du fine-tuning (LFM2, Baguettotron, Mistral, etc.).

---

## 1. Objectif

Chaque fiche (input → output) peut être décrite par des **indicateurs numériques de style** calculés sur l’**output**. En les gardant dans le Sheet puis en les injectant dans le format d’entraînement (p.ex. dans la trace `<think>`, en métadonnées, ou en conditioning), le modèle peut :

- apprendre à associer certains styles à certaines instructions ;
- être conditionné à produire un style cohérent (p.ex. TTR cible, longueur de phrase) ;
- filtrer ou pondérer les exemples (p.ex. garder les fiches avec un bon contraste syntaxique input/output).

---

## 2. Statistiques déjà présentes (cache actuel)

| Colonne | Description | Intérêt pour le fine-tuning |
|--------|-------------|-----------------------------|
| `_ratio` | Amplification (mots output / input) | Cible ou condition pour « Normalisation » vs « Expansion ». |
| `_ttr` | Type-Token Ratio (diversité lexicale) | Condition « vocabulaire riche » vs « répétitif ». |
| `_long_phrases` | Longueur moyenne des phrases (mots) | Rythme : court / équilibré / ample. |
| `_signature_json` | 7 axes (noms+adj, verbes, adv, ponct, long. mots, participes, dét. définis) | Vecteur de style dense ; peut être résumé en trace ou en conditioning. |
| `_coherence_score` | Alignement avec la moyenne du dataset (0–100) | Qualité / filtrage des exemples. |
| `_trigrams_json` | Distribution des trigrammes POS | Empreinte syntaxique ; utile pour diversité ou clustering. |

**Recommandation** : continuer à les stocker et, à l’export JSONL, les inclure dans la balise `<think>` ou en métadonnées (p.ex. une ligne de chiffres clés ou un résumé texte) pour que le modèle les voie à l’entraînement.

---

## 3. Statistiques additionnelles pertinentes (ajoutées)

Les métriques suivantes sont **pertinentes pour le fine-tuning** et ont été ajoutées au cache par ligne.

| Colonne | Description | Intérêt pour le fine-tuning |
|--------|-------------|-----------------------------|
| `_lexical_density` | (Noms + Verbes + Adj + Adv) / total tokens | « Dense » vs « fluide » : plus de contenu vs plus de mots-outils. |
| `_weak_verb_ratio` | Proportion de verbes être/avoir/faire/aller/dire parmi les verbes | Cible de style (réduire les verbes faibles = style plus précis). |
| `_syntax_contrast` | Distance stylistique input ↔ output (0–1) | Filtrage / pondération : contraste élevé = bonne transformation. |
| `_nb_sentences` | Nombre de phrases (output) | Structure et taille du texte généré. |
| `_punct_exp` | Comptage ponctuation expressive : tirets —, ..., : | Style littéraire (ponctuation expressive). |
| `_stop_ratio_out` | Proportion de mots-outils (output) | Complément de la densité lexicale ; fluidité. |

---

## 4. Comment les utiliser au fine-tuning

- **Conditioning** : ajouter dans l’instruction ou en préfixe une description courte dérivée des stats (p.ex. « TTR≈0.72, phrases ~14 mots, densité 0.58 ») pour que le modèle apprenne à lier métadonnées et style.
- **Filtrage** : exclure les fiches avec `_syntax_contrast` trop bas (transformation faible) ou `_coherence_score` trop bas.
- **Pondération** : surpondérer les exemples avec un bon contraste ou une cohérence élevée.
- **Cibles** : utiliser `_weak_verb_ratio`, `_ttr`, `_long_phrases` comme objectifs dans une loss auxiliaire ou comme critères de sélection de données.

---

## 5. Injection dans les exports

Lorsque l'option **Inclure indicateurs stylométriques** est activée dans la sidebar (Export Fine-tuning), l'app construit un résumé compact (TTR, mots/phrase, densité, nb phrases, verbes faibles) et l'injecte selon le format choisi :

| Format | Où sont injectés les indicateurs |
|--------|----------------------------------|
| **LFM2-24B-A2B** | Message `system` (optionnel) : paramètres type/forme/ton/support + ligne « Indicateurs : TTR≈… \| … mots/phrase \| … ». |
| **PleIAs/Baguettotron** | Dans la balise `<think>` : une ligne supplémentaire `Stylo: TTR≈… \| … mots/phrase \| …` après la trace forme/ton. |
| **Mistral Small Creative** | Préfixe du message `user` : `[Indicateurs stylométriques : TTR≈… \| …]` puis l'instruction et le brouillon. |

Les colonnes de cache utilisées pour ce résumé sont notamment `_ttr`, `_long_phrases`, `_lexical_density`, `_nb_sentences`, `_weak_verb_ratio` (si présentes).

---

## 6. Synthèse : colonnes de cache (après ajout)

| Colonne | Type | Rôle |
|--------|------|------|
| `_ratio` | float | Amplification |
| `_ttr` | float | Diversité lexicale |
| `_long_phrases` | float | Longueur moyenne des phrases |
| `_signature_json` | JSON | 7 axes stylométriques |
| `_coherence_score` | int 0–100 | Cohérence avec le dataset |
| `_trigrams_json` | JSON | Trigrammes POS |
| `_lexical_density` | float | Densité lexicale (nouveau) |
| `_weak_verb_ratio` | float | Part des verbes faibles (nouveau) |
| `_syntax_contrast` | float | Contraste input/output (nouveau) |
| `_nb_sentences` | int | Nombre de phrases (nouveau) |
| `_punct_exp` | str "n,m,p" | Ponctuation expressive —, ..., : (nouveau) |
| `_stop_ratio_out` | float | Part des mots-outils (nouveau) |

Toutes sont calculées automatiquement à la sauvegarde (après « Vérifier ma prose » + « Enregistrer »).
