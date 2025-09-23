# Implémentation d'un Réseau de Neurones de fanorontelo (jeu malagasy)

Ce dépôt contient une implémentation en Rust d'un réseau de neurones, conçu pour trouver le meilleur coup d'une position de fanorontelo (jeu de strategie malagasy).

## Bibliothèques Utilisées
- **Rayon** : Bibliothèque Rust pour le parallélisme des données, utilisée pour accélérer les calculs sur les vecteurs et matrices.
- **Utils (interne)** : Module personnalisé contenant des fonctions utilitaires pour l'initialisation des matrices/vecteurs, les produits matrice-vecteur, et les fonctions sigmoïde et softmax.

## Documentation Mathématique
- **Fonction de perte** : Log-vraisemblance négative : `Perte = -log(a_{L-1}[d*]) - log(a_{L-1}[a*])`.
- **Mise à jour des paramètres** : Descente de gradient avec `W_k -= lr * ∂Perte/∂W_k` et `b_k -= lr * ∂Perte/∂b_k`.
- **Prédiction** : Sortie `((d*, pd*), (a*, pa*))`, où `d*` et `a*` sont les indices des probabilités maximales pour chaque groupe de sortie, avec `pd*` et `pa*` leurs probabilités.

### Caractéristiques Principales
- **Activation** : Sigmoïde pour les couches cachées, softmax pour la sortie (deux groupes).
- **Objectif** : Classification multi-classes avec deux groupes de sorties.
- **Optimisation** : Descente de gradient avec parallélisation via `rayon`.

Cette implémentation est conçue pour une tâche de classification avec deux groupes de sorties, exploitant les activations sigmoïdes et softmax pour gérer les relations non linéaires.
