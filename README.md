# Implémentation d'un Réseau de Neurones de fanorontelo (jeu malagasy)

Ce dépôt contient une implémentation en Rust d'un réseau de neurones, conçu pour trouver le meilleur coup d'une position de fanorontelo (jeu de strategie malagasy).

## Bibliothèques Utilisées
- **Rayon** : Bibliothèque Rust pour le parallélisme des données, utilisée pour accélérer les calculs sur les vecteurs et matrices.
- **Utils (interne)** : Module personnalisé contenant des fonctions utilitaires pour l'initialisation des matrices/vecteurs, les produits matrice-vecteur, et les fonctions sigmoïde et softmax.

## Documentation Mathématique
### Notation
- **L** : Nombre de couches, incluant les couches d'entrée, cachées et de sortie.
- **ls** : Vecteur des tailles des couches, où `ls[k]` est le nombre de neurones dans la couche `k`.
- **is** : Taille de l'entrée (nombre de caractéristiques dans le vecteur d'entrée).
- **W_k** : Matrice de poids pour la couche `k`, de taille `ls[k] × ls[k-1]` (ou `ls[k] × is` pour la couche d'entrée).
- **b_k** : Vecteur de biais pour la couche `k`, de taille `ls[k]`.
- **lr** : Taux d'apprentissage, contrôlant la taille du pas dans les mises à jour des paramètres.
- **x** : Vecteur d'entrée de taille `is`.
- **z_k** : Valeurs de pré-activation pour la couche `k`, calculées comme `z_k = W_k * a_{k-1} + b_k`.
- **a_k** : Valeurs d'activation pour la couche `k`, où `a_k = σ(z_k)` pour les couches cachées (sigmoïde) et `a_L = softmax(z_L)` pour la couche de sortie.
- **d* et a*** : Étiquettes cibles pour les deux groupes de sortie (par exemple, un digit et un autre attribut).
- **σ(u)** : Fonction sigmoïde, définie comme `σ(u) = 1 / (1 + e^(-u))`.
- **softmax(z)** : Fonction softmax, définie comme `softmax(z)_i = e^(z_i) / Σ_j e^(z_j)`.

### Modèle
- **Entrée**: Vecteur contenant les informations de la position (chaque point est codé en one-hot), à la fin on ajoute la représentaion du joueur actuelle.
- **Structure** : Réseau à `L` couches avec `ls[0]` neurones en entrée, `ls[1]` à `ls[L-2]` pour les couches cachées, et `ls[L-1]` pour la sortie.
- **Initialisation** : Poids `W_k` initialisés aléatoirement, biais `b_k` à zéro ou petites valeurs, avec un taux d'apprentissage `lr`.
- **Propagation avant** :
  - Couches cachées : `a_k = σ(W_k * a_{k-1} + b_k)` avec activation sigmoïde.
  - Couche de sortie : `z_{L-1}` divisé en deux parties, chacune transformée par softmax, puis concaténée : `a_{L-1} = [softmax(z_{L-1}[0..9]), softmax(z_{L-1}[9..])]`.
- **Fonction de perte** : Log-vraisemblance négative : `Perte = -log(a_{L-1}[d*]) - log(a_{L-1}[a*])`.
- **Mise à jour des paramètres** : Descente de gradient avec `W_k -= lr * ∂Perte/∂W_k` et `b_k -= lr * ∂Perte/∂b_k`.
- **Prédiction** : Sortie `((d*, pd*), (a*, pa*))`, où `d*` et `a*` sont les indices des probabilités maximales pour chaque groupe de sortie, avec `pd*` et `pa*` leurs probabilités.

### Caractéristiques Principales
- **Activation** : Sigmoïde pour les couches cachées, softmax pour la sortie (deux groupes).
- **Objectif** : Classification multi-classes avec deux groupes de sorties.
- **Optimisation** : Descente de gradient avec parallélisation via `rayon`.

Cette implémentation est conçue pour une tâche de classification avec deux groupes de sorties, exploitant les activations sigmoïdes et softmax pour gérer les relations non linéaires.
