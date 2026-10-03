# Lab 3 – Introduction to Machine Learning

**Auteure :** Dorline TIAKO OKASSIE — ING4 Data & Intelligence Artificielle, ECE Paris (Groupe 01)
**Cours :** Introduction to Machine Learning
**Date :** octobre 2026

Ce lab contient quatre exercices. Les trois premiers consistent à **réimplémenter « from scratch »** des outils classiques de scikit-learn : la validation croisée k-fold, les métriques de classification et la recherche aléatoire d'hyperparamètres. Le quatrième est un **projet complet de régression** : prédire le prix de voitures BMW d'occasion.

---

## Structure du dépôt

```
.
├── Exo1_k_fold.ipynb                          # Validation croisée k-fold from scratch
├── Exo2_classification_metrics.ipynb          # Precision, Recall, F1 from scratch
├── Exo3_random_search.ipynb                   # Random search from scratch
├── Exo4_bmw_price_prediction_exercise_en.ipynb # Prédiction du prix des BMW (régression)
├── bmw_pricing_challenge.csv                  # Données de l'exercice 4
├── requirements.txt
└── README.md
```

## Lancer les notebooks

```bash
pip install -r requirements.txt
jupyter lab
```

Ouvrez ensuite un notebook et faites **Kernel → Restart Kernel and Run All Cells**.
Le fichier `bmw_pricing_challenge.csv` peut être placé à côté du notebook ou dans un dossier `data/` : le notebook le trouve automatiquement.

---

## Exercice 1 – Validation croisée k-fold (from scratch)

### Objectif
Écrire une fonction `k_fold_cross_validation(X, y, k, model)` qui évalue un modèle par validation croisée k-fold aléatoire, sans utiliser `KFold` ni `cross_val_score`. Le jeu de données utilisé est **Iris** (150 fleurs, 4 variables, 3 classes).

### Problèmes rencontrés
- **Les données d'Iris sont triées par classe** : 50 setosa, puis 50 versicolor, puis 50 virginica. Un découpage sans mélange donnerait des folds contenant une seule classe, et le modèle serait testé sur une classe qu'il n'a presque jamais vue.
- Le nombre de lignes n'est pas toujours divisible par `k`.

### Solution
1. On mélange les indices avec `np.random.default_rng(...).permutation`. La graine est fixée pour rendre les résultats reproductibles.
2. On découpe les données en `k` folds avec `np.array_split`, qui gère les tailles inégales.
3. On fait une boucle sur les folds. À chaque tour, le fold *i* sert de jeu de test et les `k-1` autres sont concaténés pour former le jeu d'entraînement.
4. On entraîne le modèle, on calcule l'accuracy (`np.mean(y_pred == y_test)`), puis on renvoie la moyenne des accuracies.

### Résultats (Random Forest, k = 5)

| Fold | 1 | 2 | 3 | 4 | 5 | **Moyenne** |
|---|---|---|---|---|---|---|
| Accuracy | 0.933 | 0.967 | 0.967 | 0.933 | 0.967 | **0.953 ± 0.016** |

Pour vérifier, scikit-learn (`KFold` + `cross_val_score`) donne **0.960**. L'écart vient uniquement du mélange aléatoire, qui produit des folds différents. Avec k = 3, 5 et 10, notre implémentation donne toujours environ 0.953.

---

## Exercice 2 – Métriques de classification (from scratch)

### Objectif
Implémenter la **précision**, le **rappel** et le **F1-score** sans aucune bibliothèque de métriques. Le jeu de données est **Breast Cancer** (569 tumeurs, 30 variables). Le modèle est une régression logistique.

### Problèmes rencontrés
- Il faut d'abord compter correctement les **TP, FP, FN** (et TN).
- Il y a un risque de **division par zéro**, par exemple si le modèle ne prédit jamais la classe positive.
- La régression logistique affichait un avertissement de **non-convergence**.

### Solution
- Une fonction `confusion_counts()` compte TP, FP, FN et TN avec des masques booléens NumPy. La classe positive est 1 (`benign`).
- Les formules utilisées :
  - Précision = TP / (TP + FP)
  - Rappel = TP / (TP + FN)
  - F1 = 2 · P · R / (P + R)
- Chaque fonction renvoie 0 quand le dénominateur est nul.
- `max_iter=10000` sur `LogisticRegression` supprime l'avertissement de convergence.

### Résultats (jeu de test, 114 tumeurs)

| TP | FP | FN | TN |
|---|---|---|---|
| 62 | 1 | 5 | 46 |

| Métrique | Notre implémentation | scikit-learn |
|---|---|---|
| Précision | **0.9841** | 0.9841 |
| Rappel | **0.9254** | 0.9254 |
| F1-score | **0.9538** | 0.9538 |

Les valeurs sont **identiques à scikit-learn**. scikit-learn n'est utilisé que pour vérifier les résultats.

---

## Exercice 3 – Random Search (from scratch)

### Objectif
Coder une recherche aléatoire d'hyperparamètres pour un `DecisionTreeClassifier`, sans `RandomizedSearchCV`. La grille de recherche est la suivante :

| Hyperparamètre | Valeurs |
|---|---|
| `max_depth` | 3, 5, 7, None |
| `criterion` | gini, entropy |
| `min_samples_split` | 2, 3, 4, 5 |

### Problèmes rencontrés
- Un tirage naïf peut **tirer deux fois la même combinaison**.
- Évaluer chaque combinaison sur le jeu de test introduirait un **biais** : on choisirait les paramètres en regardant le test.
- Un même objet modèle réutilisé d'une itération à l'autre peut garder des paramètres de l'essai précédent.

### Solution
1. On génère les 32 combinaisons possibles avec `itertools.product`.
2. On en tire `n_iter = 10` **sans remise** (`rng.choice(..., replace=False)`).
3. Chaque combinaison est évaluée par une **validation croisée 5-fold codée à la main**, uniquement sur le jeu d'entraînement.
4. Chaque essai utilise une copie neuve du modèle (`clone`).
5. On garde la combinaison qui a la meilleure accuracy moyenne, puis on réentraîne le modèle sur tout le train.

### Résultats

- **Meilleurs paramètres :** `max_depth=3`, `criterion='gini'`, `min_samples_split=4`
- **Accuracy en validation croisée :** 0.936

| Modèle | Accuracy sur le test |
|---|---|
| Arbre par défaut | 0.912 |
| **Arbre optimisé (random search)** | **0.965** |

Limiter la profondeur à 3 réduit le sur-apprentissage et fait gagner **environ 5 points** d'accuracy. Il a suffi de 10 essais au lieu des 32 d'une grid search.

---

## Exercice 4 – Prédiction du prix des BMW d'occasion

### Objectif
Prédire le prix de vente (enchère B2B, 2018) de 4 843 BMW d'occasion à partir de plusieurs variables :
- kilométrage et puissance ;
- dates d'immatriculation et de vente ;
- carburant, couleur, modèle et type de carrosserie ;
- 8 options d'équipement (`feature_1` à `feature_8`).

On compare ensuite trois modèles linéaires (**Linear Regression, Ridge, Lasso**) avec le **R²** comme métrique.

### Démarche
1. **Analyse exploratoire :** valeurs manquantes, distributions du prix et des variables numériques, histogrammes des variables catégorielles.
2. **Séparation train / test (80 / 20)**, faite *avant* toute transformation.
3. **Feature engineering :**
   - suppression des variables inutiles ;
   - création de l'âge du véhicule ;
   - polynômes de degré 2 ;
   - standardisation ;
   - one-hot encoding ;
   - conversion des booléens en 0/1.
4. **Modélisation :** Linear Regression, puis Ridge et Lasso avec α choisi par validation croisée 5-fold (`RidgeCV`, `LassoCV`).
5. **Interprétation :** graphique des 25 poids les plus importants pour chaque modèle.

### Problèmes rencontrés et solutions

| Problème | Solution |
|---|---|
| Valeurs impossibles : kilométrage négatif (1), puissance = 0 (1), prix < 1 000 € (62) | Suppression de ces 64 lignes (4 843 → 4 779 voitures) |
| `maker_key` vaut toujours « BMW » | Variable supprimée, car elle n'apporte aucune information |
| Dates brutes (`registration_date`, `sold_at`) inutilisables telles quelles | Création de `car_age` = âge de la voiture (en années) au moment de la vente |
| Relations non linéaires (prix ↔ âge, kilométrage, puissance) | `PolynomialFeatures(degree=2)` sur les trois variables numériques |
| Échelles très différentes (km ≈ 10⁵, puissance ≈ 10²) | `StandardScaler`, indispensable pour Ridge et Lasso |
| Catégories sans ordre naturel, `model_key` avec beaucoup de modèles rares | `OneHotEncoder` avec `min_frequency=10`. Les modèles rares (ou absents du train) sont regroupés dans une catégorie « infrequent » |
| Risque de **fuite de données** (data leakage) | Tous les transformateurs sont ajustés (`fit`) sur le train seulement, puis appliqués (`transform`) au test |
| Prix très asymétrique (skewness = 3.47) | Étudié dans l'EDA (comparaison avec log(prix)). Le prix brut est conservé, comme demandé par l'énoncé |
| `FileNotFoundError` sur `data/bmw_pricing_challenge.csv` | Le notebook cherche maintenant le CSV automatiquement (dossier courant, `data/` ou dossier parent) |

Le jeu final contient **69 variables explicatives** : 9 numériques polynomiales, 52 colonnes one-hot et 8 booléennes. Il y a **3 823 voitures en train** et **956 en test**.

### Résultats finaux (jeu de test)

| Modèle | α (CV) | R² test | RMSE test (€) |
|---|---|---|---|
| Linear Regression | – | 0.8479 | 3 368 |
| Ridge | 2.85 | 0.8477 | 3 371 |
| **Lasso** | 2.36 | **0.8487** | **3 359** |

### Interprétation
- Les trois modèles expliquent environ **85 % de la variance du prix**, avec une erreur moyenne d'environ 3 400 €.
- Les variables les plus influentes sont la **puissance**, l'**âge** et le **kilométrage** (avec leurs termes polynomiaux), ainsi que certains **modèles spécifiques** comme les séries X et M.
- **Ridge** réduit tous les poids. Le modèle est plus stable quand les variables sont corrélées (polynômes, one-hot).
- **Lasso** met **10 poids sur 69 exactement à 0**. Il fait donc aussi de la sélection de variables, et c'est le modèle le plus simple et le plus performant ici.
- Le R² du train (0.77) est plus bas que celui du test (0.85). Cela s'explique par quelques voitures très chères (jusqu'à 178 500 €) restées dans le jeu d'entraînement, que les modèles linéaires prédisent mal.

### Pistes d'amélioration
- Prédire `log(prix)` pour réduire l'effet de l'asymétrie et des valeurs extrêmes.
- Tester des modèles non linéaires (Random Forest, Gradient Boosting), déjà importés dans le notebook.

---

## Synthèse

| Exercice | Ce qui a été réalisé | Résultat clé |
|---|---|---|
| 1 – K-fold | Validation croisée codée à la main | Accuracy moyenne **0.953** (sklearn : 0.960) |
| 2 – Métriques | Précision / Rappel / F1 à partir de TP, FP, FN | **P = 0.984, R = 0.925, F1 = 0.954**, identiques à sklearn |
| 3 – Random search | Recherche aléatoire + CV maison | Test **0.965** vs 0.912 sans réglage |
| 4 – Prix BMW | Pipeline complet + 3 modèles linéaires | **R² = 0.849** (Lasso) |

## Technologies
Python 3 · NumPy · pandas · scikit-learn · Matplotlib · Seaborn · Jupyter
