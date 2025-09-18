# Implémentation d'un Réseau de Neurones

Ce dépôt contient une implémentation en Rust d'un réseau de neurones à propagation avant avec rétropropagation, conçu pour une tâche de classification avec deux groupes de sorties softmax. L'implémentation utilise la crate `rayon` pour des calculs parallélisés et inclut des fonctions utilitaires pour les opérations sur les matrices et les vecteurs.

## Documentation Mathématique

### Notation
- **L** : Nombre de couches (`ln`), incluant les couches d'entrée, cachées et de sortie.
- **ls** : Vecteur des tailles des couches, où `ls[k]` est le nombre de neurones dans la couche `k`.
- **is** : Taille de l'entrée (nombre de caractéristiques dans le vecteur d'entrée).
- **W_k** : Matrice de poids pour la couche `k`, où `W_k[i][j]` connecte le neurone `j` de la couche `k-1` au neurone `i` de la couche `k`.
- **b_k** : Vecteur de biais pour la couche `k`.
- **lr** : Taux d'apprentissage (`lr`), contrôlant la taille du pas dans les mises à jour des paramètres.
- **x** : Vecteur d'entrée de taille `is`.
- **z_k** : Valeurs de pré-activation pour la couche `k`, calculées comme `z_k = W_k * a_{k-1} + b_k`.
- **a_k** : Valeurs d'activation pour la couche `k`, où `a_k = σ(z_k)` pour les couches cachées (activation sigmoïde) et `a_L = softmax(z_L)` pour la couche de sortie.
- **d* et a*** : Étiquettes cibles pour les deux groupes de sortie (par exemple, un digit et un autre attribut).
- **σ(u)** : Fonction sigmoïde, définie comme `σ(u) = 1 / (1 + e^(-u))`.
- **softmax(z)** : Fonction softmax pour un vecteur `z`, définie comme `softmax(z)_i = e^(z_i) / Σ_j e^(z_j)`.

### Initialisation
Le réseau de neurones est initialisé avec :
- Un vecteur de tailles de couches `ls` (incluant les couches d'entrée et de sortie).
- La taille de l'entrée `is`.
- Le taux d'apprentissage `lr`.

Les poids `W_k` et les biais `b_k` sont initialisés à l'aide des fonctions utilitaires (`init_matrixes` et `init_vectors`). Les poids sont généralement initialisés aléatoirement, et les biais sont définis à zéro ou à de petites valeurs.

#### Représentation Mathématique
- **Poids** : Pour la couche `k`, `W_k` est une matrice de taille `ls[k] × ls[k-1]` (ou `ls[k] × is` pour la couche d'entrée).
- **Biais** : Pour la couche `k`, `b_k` est un vecteur de taille `ls[k]`.
- **Structure** : Le réseau a `L = ls.len()` couches, avec `ls[0]` neurones dans la couche d'entrée, `ls[1]` à `ls[L-2]` pour les couches cachées, et `ls[L-1]` pour la couche de sortie.

### Propagation Avant
La propagation avant calcule la sortie du réseau pour un vecteur d'entrée `x`.

#### Étapes
1. **Couche d'entrée (k = 0)** :
   - Calculer la pré-activation : `z_0 = W_0 * x + b_0`.
   - Appliquer l'activation sigmoïde : `a_0 = σ(z_0)`, où `σ(u) = 1 / (1 + e^(-u))` est appliquée élément par élément.
2. **Couches cachées (k = 1 à L-2)** :
   - Pour chaque couche `k` :
     - Calculer la pré-activation : `z_k = W_k * a_{k-1} + b_k`.
     - Appliquer l'activation sigmoïde : `a_k = σ(z_k)`.
3. **Couche de sortie (k = L-1)** :
   - Calculer la pré-activation : `z_{L-1} = W_{L-1} * a_{L-2} + b_{L-1}`.
   - Diviser `z_{L-1}` en deux parties : `z_{L-1}[0..9]` (les 9 premiers éléments) et `z_{L-1}[9..]` (les éléments restants).
   - Appliquer la fonction softmax à chaque partie :
     - `sf1 = softmax(z_{L-1}[0..9])`.
     - `sf2 = softmax(z_{L-1}[9..])`.
   - Concaténer les résultats : `a_{L-1} = [sf1, sf2]`.

#### Sortie
La sortie `a_{L-1}` est une distribution de probabilité sur deux groupes de classes, avec `sf1` et `sf2` représentant les probabilités normalisées pour chaque groupe.

### Rétropropagation
La rétropropagation calcule les gradients de la fonction de perte par rapport aux paramètres `W_k` et `b_k` et les met à jour à l'aide de la descente de gradient.

#### Fonction de Perte
La perte est la log-vraisemblance négative des probabilités prédites pour les étiquettes vraies `d*` et `a*` :
- `Perte = -log(P_d(d*) * P_a(a*)) = -log(a_{L-1}[d*]) - log(a_{L-1}[a*])`.

#### Étapes
1. **Gradient de la couche de sortie** :
   - Initialiser le gradient `dz_{L-1}` :
     - Pour l'index `i` dans `a_{L-1}` :
       - Si `i == d*`, définir `dz_{L-1}[i] = a_{L-1}[i] - 1`.
       - Si `i == a*`, définir `dz_{L-1}[i] = a_{L-1}[i] - 1`.
       - Sinon, `dz_{L-1}[i] = a_{L-1}[i]`.
   - Cela correspond à la dérivée de la perte par rapport à `z_{L-1}` pour les sorties softmax.
2. **Gradient des couches cachées** :
   - Pour chaque couche `k` de `L-2` à `0` :
     - Calculer les gradients des poids : `dw_k[i][j] = dz_k[i] * a_{k-1}[j]`, où `a_{-1} = x` pour la couche d'entrée.
     - Calculer les gradients des biais : `db_k[i] = dz_k[i]`.
     - Si `k > 0`, calculer le gradient d'activation pour la couche précédente :
       - `da_{k-1}[i] = Σ_j (dz_k[j] * W_k[j][i])`.
       - `dz_{k-1}[i] = da_{k-1}[i] * a_{k-1}[i] * (1 - a_{k-1}[i])` (dérivée de la sigmoïde).
3. **Mise à jour des paramètres** :
   - Mettre à jour les poids : `W_k[i][j] -= lr * dw_k[i][j]`.
   - Mettre à jour les biais : `b_k[i] -= lr * dz_k[i]`.

### Parallélisation
L'implémentation utilise `rayon` pour l'itération parallèle (par exemple, dans l'application de la sigmoïde), améliorant les performances pour les grands vecteurs.

### Prédiction
La fonction `predict` retourne les classes les plus probables et leurs probabilités :
- Exécuter la propagation avant pour obtenir `a_{L-1}`.
- Pour le premier groupe (`sf[0..9]`) :
  - Trouver `d* = argmax(sf[0..9])` et sa probabilité `pd* = max(sf[0..9])`.
- Pour le second groupe (`sf[9..]`) :
  - Trouver `a* = argmax(sf[9..])` et sa probabilité `pa* = max(sf[9..])`.
- Retourner `((d*, pd*), (a*, pa*))`.

### Caractéristiques Principales
- **Activation Sigmoïde** : Utilisée pour les couches cachées pour introduire une non-linéarité.
- **Sortie Softmax** : Normalise les sorties en probabilités pour deux groupes de classes.
- **Rétropropagation** : Calcule les gradients efficacement avec des opérations parallélisées.
- **Descente de Gradient** : Met à jour les paramètres en utilisant le taux d'apprentissage `lr`.

Cette implémentation est conçue pour une tâche de classification avec deux groupes de sorties, utilisant des activations sigmoïdes et softmax pour gérer les relations non linéaires et la classification multi-classes.
