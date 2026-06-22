# Simulation Météo Distribuée avec GPU

Projet de Système Réparti : Calcul parallèle (GPU/CuPy) et distribué (MPI).

## Membres de l'équipe et Responsabilités Techniques
* **Malak Boussetta** : Infrastructure de communication distribuée (MPI) et Décomposition de domaine.
* **Aya** : Développement des cœurs de calcul parallèle et optimisation sur GPU (CuPy/CUDA).
* **Marwa** : Gestion de l'échange réseau des bordures (Zones Tampons / Halos).

## Architecture du Projet
* `src/` : Code source et modules distribués du système.
* `tests/` : Scripts de validation de la simulation thermique.
