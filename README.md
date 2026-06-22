# Simulation Climatique Distribuée (MPI / GPU)

Ce projet implémente une simulation de diffusion thermique pour des modèles climatiques en utilisant une architecture distribuée et parallélisée. Il combine la décomposition de domaine via **MPI** et l'accélération matérielle par **GPU** (CuPy).

## 👥 L'Équipe & Rôles
- **Malak** : Scrum Master, Infrastructure MPI (`MPIManager`), décomposition de domaine et script d'orchestration principal.
- **Aya** : Développement du Kernel de calcul de diffusion sur GPU (`ClimateSimulationKernel`).
- **Marwa** : Product Owner / QA, Gestion de l'échange de frontières réseau (`halo_exchange`).

## 🛠️ Architecture du Code
Le projet est structuré de la manière suivante dans le dossier `src/` :
- `mpi_manager.py` : Gère la topologie des processus, le découpage en bandes horizontales et la centralisation des résultats.
- `simulation_kernel.py` : Implémente le stencil de calcul sur le GPU via CuPy.
- `halo_exchange.py` : Assure la communication MPI des lignes fantômes (halos) entre les processus voisins.
- `main_simulation.py` : Orchestre la boucle temporelle, l'échange de halos et les calculs.

## 🚀 Exécution
Pour lancer la simulation répartie sur 3 processus virtuels dans l'environnement configuré :
```bash
PYTHONPATH=src mpirun --allow-run-as-root --oversubscribe -np 3 python3 src/main_simulation.py
