# Projet Systèmes Répartis — Simulation Météo Distribuée avec GPU

**Sujet 11 — Guide pratique de reproduction**

Master d'Excellence en Intelligence Artificielle
Faculté des Sciences Ben M'Sik — Université Hassan II de Casablanca

---

## Sommaire

1. [Présentation du projet](#1-présentation-du-projet)
2. [Les trois piliers de l'architecture](#2-les-trois-piliers-de-larchitecture)
3. [Prérequis](#3-prérequis)
4. [Structure du projet](#4-structure-du-projet)
5. [Étape 1 — Installation de l'environnement](#5-étape-1--installation-de-lenvironnement)
6. [Étape 2 — La partition de la grille (découpage spatial)](#6-étape-2--la-partition-de-la-grille-découpage-spatial)
7. [Étape 3 — Le calcul GPU avec CuPy](#7-étape-3--le-calcul-gpu-avec-cupy)
8. [Étape 4 — La communication des frontières (MPI / Halos)](#8-étape-4--la-communication-des-frontières-mpi--halos)
9. [Étape 5 — Orchestration et lancement de la simulation](#9-étape-5--orchestration-et-lancement-de-la-simulation)
10. [Résultats attendus](#10-résultats-attendus)
11. [SOS / Dépannage](#11-sos--dépannage)
12. [Auteurs](#12-auteurs)

---

## 1. Présentation du projet

Ce projet implémente une **simulation de diffusion thermique** pour des modèles météorologiques en combinant deux technologies de calcul haute performance :

- **MPI** (*Message Passing Interface*) : découpe la carte météo en bandes horizontales et distribue chaque bande sur un processus indépendant.
- **CuPy / CUDA** : exécute, sur le GPU de chaque nœud, le calcul mathématique (équation de diffusion de la chaleur) de façon massivement parallèle.

L'idée centrale est simple : une carte météo mondiale est trop grande pour être calculée par un seul ordinateur en temps raisonnable. On la **découpe**, on distribue chaque morceau sur un nœud, et on **synchronise les bords** (halos) à chaque pas de temps pour que la simulation reste cohérente globalement.

> 💡 **Analogie pédagogique :** Imaginez une carte de France découpée en bandes horizontales — une bande par processus. Chaque processus calcule la diffusion de chaleur sur sa bande. Mais les bords entre les bandes doivent être mis à jour à chaque étape : c'est l'échange de halos.

---

## 2. Les trois piliers de l'architecture

| Pilier | Technologie | Fichier source | Rôle |
|--------|------------|----------------|------|
| **Partition de la grille** | MPI / NumPy | `src/mpi_manager.py` | Découpe la grille globale en bandes horizontales et les distribue |
| **Calcul GPU** | CuPy / CUDA | `src/simulation_kernel.py` | Applique l'équation de diffusion sur GPU, en parallèle |
| **Communication des frontières** | MPI / Halos | `src/halo_exchange.py` | Échange les lignes de bord entre processus voisins |

Ces trois composants sont orchestrés par `src/main_simulation.py`.

---

## 3. Prérequis

### 3.1 Matériel

- Un **GPU NVIDIA** compatible CUDA (ou Google Colab avec GPU T4 activé)
- Au moins **4 Go de RAM GPU**

### 3.2 Logiciel

- **Python** ≥ 3.8
- **Linux** (ou Google Colab) — MPI ne fonctionne pas nativement sous Windows ; utilisez WSL 2 Ubuntu si vous êtes sous Windows
- Compte **Google** (pour Colab)

### 3.3 Pourquoi Google Colab ?

Google Colab offre gratuitement un GPU NVIDIA T4 et un environnement Linux préconfiguré. C'est l'environnement recommandé pour ce projet si vous n'avez pas de machine Linux avec GPU en local.

> ⚠️ **Important :** Avant de commencer, vérifiez que votre session Colab est bien configurée avec un GPU :
> `Exécution → Modifier le type d'exécution → Accélérateur matériel : GPU (T4)`

---

## 4. Structure du projet

```
simulation-meteo-distribuee/
├── src/
│   ├── mpi_manager.py          # Pilier 1 : partition de la grille
│   ├── simulation_kernel.py    # Pilier 2 : calcul GPU (CuPy)
│   ├── halo_exchange.py        # Pilier 3 : communication des frontières
│   └── main_simulation.py      # Orchestrateur principal
├── tests/                      # Tests de validation
└── README.md
```

---

## 5. Étape 1 — Installation de l'environnement

Cette étape installe les trois dépendances critiques du projet. Exécutez ces cellules **dans l'ordre** dans votre notebook Colab.

### 5.1 Installer OpenMPI et mpi4py

MPI est un standard de communication entre processus distribués. `openmpi-bin` est la bibliothèque système (C/C++), et `mpi4py` est l'interface Python qui l'enveloppe.

```python
# 1. Mise à jour et installation du protocole MPI pour Linux
!apt-get update && apt-get install -y openmpi-bin libopenmpi-dev

# 2. Installation de l'extension Python pour MPI
!pip install mpi4py
```

**Pourquoi ces deux commandes ?** `apt-get` installe les binaires MPI au niveau du système (exécutables `mpirun`, bibliothèques partagées). `pip install mpi4py` installe ensuite le binding Python qui permet d'appeler ces bibliothèques depuis vos scripts.

### 5.2 Installer CuPy

CuPy est l'équivalent GPU de NumPy. Il permet d'écrire du code Python presque identique à NumPy, mais les calculs s'exécutent sur le GPU via CUDA.

```python
# 3. Installation de CuPy adapté à la version CUDA de Colab
!pip install cupy-cuda12x
```

**Pourquoi `cupy-cuda12x` ?** Colab utilise CUDA 12.x. Le suffixe `cuda12x` indique à pip d'installer la version compilée pour cette version précise de CUDA. Si vous utilisez une autre version de CUDA (11.x par exemple), remplacez par `cupy-cuda11x`.

### 5.3 Vérification de l'installation

Après les installations, vérifiez que tout fonctionne :

```python
# Test MPI
from mpi4py import MPI
print("mpi4py OK — version MPI :", MPI.Get_version())

# Test CuPy
import cupy as cp
print("CuPy OK — GPU détecté :", cp.cuda.runtime.getDeviceCount(), "GPU(s)")
print("Nom du GPU :", cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
```

**Sortie attendue :**
```
mpi4py OK — version MPI : (3, 1)
CuPy OK — GPU détecté : 1 GPU(s)
Nom du GPU : Tesla T4
```

> 🆘 Si vous obtenez `CuPy OK — GPU détecté : 0 GPU(s)`, votre session Colab n'a pas de GPU activé. Allez dans `Exécution → Modifier le type d'exécution` et sélectionnez **GPU T4**.

### 5.4 Créer la structure de dossiers

```python
# Création des dossiers du projet
!mkdir -p simulation-meteo-distribuee/src simulation-meteo-distribuee/tests
%cd simulation-meteo-distribuee
```

**Pourquoi `%cd` et non `!cd` ?** En Colab, `!cd` crée un sous-shell temporaire qui disparaît aussitôt. `%cd` est une commande "magique" IPython qui change réellement le répertoire courant de votre session.

---

## 6. Étape 2 — La partition de la grille (découpage spatial)

### 6.1 Concept : la décomposition de domaine

La grille météorologique globale (par exemple 120×120 cellules) est trop large pour être traitée par un seul processus. On la découpe en **bandes horizontales** — une bande par processus MPI.

```
Grille globale 120×120
┌──────────────────────┐
│  Processus 0 (rang 0)│  ← lignes 0 à 39
├──────────────────────┤
│  Processus 1 (rang 1)│  ← lignes 40 à 79
├──────────────────────┤
│  Processus 2 (rang 2)│  ← lignes 80 à 119
└──────────────────────┘
```

Chaque processus ne "voit" et ne calcule que sa propre bande. Mais pour que la physique soit correcte aux bords, il doit connaître la dernière ligne de son voisin du dessus et la première ligne de son voisin du dessous → c'est le rôle des **halos** (Étape 4).

### 6.2 Créer le fichier `mpi_manager.py`

```python
%%writefile src/mpi_manager.py
from mpi4py import MPI
import numpy as np

class MPIManager:
    def __init__(self, global_width, global_height):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()  # ID du nœud actuel (0, 1, 2...)
        self.size = self.comm.Get_size()  # Nombre total de nœuds

        self.global_width = global_width
        self.global_height = global_height

        # Découpage de la grille en bandes horizontales
        self.local_height = global_height // self.size
        if self.rank == self.size - 1:
            self.local_height += global_height % self.size

        self.local_width = global_width

        # Identification des voisins (Haut et Bas) pour l'échange des frontières
        self.up_neighbor = self.rank - 1 if self.rank > 0 else None
        self.down_neighbor = self.rank + 1 if self.rank < self.size - 1 else None

    def scatter_grid(self, global_grid):
        """Découpe et distribue la grille globale depuis le nœud 0 vers tous les nœuds."""
        if self.rank == 0:
            chunks = np.array_split(global_grid, self.size, axis=0)
        else:
            chunks = None
        return self.comm.scatter(chunks, root=0)

    def gather_grid(self, local_grid):
        """Rassemble les morceaux locaux pour reconstruire la grille globale sur le nœud 0."""
        gathered_chunks = self.comm.gather(local_grid, root=0)
        if self.rank == 0:
            return np.vstack(gathered_chunks)
        return None
```

### 6.3 Explication ligne par ligne

| Code | Ce que ça fait | Pourquoi c'est nécessaire |
|------|----------------|--------------------------|
| `MPI.COMM_WORLD` | Récupère le "monde" de communication, c'est-à-dire tous les processus lancés | Point d'entrée obligatoire de toute application MPI |
| `Get_rank()` | Renvoie l'ID unique de ce processus (0, 1, 2…) | Permet à chaque processus de savoir "qui il est" et d'agir différemment |
| `Get_size()` | Renvoie le nombre total de processus lancés | Permet de calculer la taille des bandes |
| `global_height // self.size` | Division entière : taille de chaque bande | Le dernier processus prend le reste (`%`) si la division n'est pas exacte |
| `up_neighbor / down_neighbor` | Identifie les voisins immédiats | Indispensable pour savoir à qui envoyer/recevoir les halos |
| `scatter_grid()` | Découpe et envoie un morceau de la grille à chaque processus | Le processus maître (rang 0) distribue le travail |
| `gather_grid()` | Collecte tous les morceaux et reconstruit la grille globale | Permet de récupérer le résultat final sur le processus maître |

---

## 7. Étape 3 — Le calcul GPU avec CuPy

### 7.1 Concept : la diffusion thermique et le stencil 2D

L'équation de diffusion de la chaleur discrétisée sur une grille 2D s'écrit :

```
T(i,j)_suivant = T(i,j) + α × (T(i-1,j) + T(i+1,j) + T(i,j-1) + T(i,j+1) - 4×T(i,j))
```

Pour chaque cellule `(i, j)` de la grille, la nouvelle température est calculée en fonction de ses **4 voisins directs** (haut, bas, gauche, droite). On appelle cela un **stencil** de diffusion.

Le GPU excelle à ce type de calcul car il peut traiter **toutes les cellules simultanément** (des milliers en parallèle), là où un CPU les traiterait en boucle une par une.

### 7.2 Créer le fichier `simulation_kernel.py`

```python
%%writefile src/simulation_kernel.py
import cupy as cp

class ClimateSimulationKernel:
    def __init__(self, local_grid_shape, alpha=0.1):
        self.alpha = alpha
        self.height, self.width = local_grid_shape

    def compute_next_step(self, d_local_grid):
        """Calcule l'état suivant de la météo (diffusion thermique) sur GPU."""
        d_next_grid = cp.copy(d_local_grid)

        # Stencil 2D de diffusion : calcul simultané sur tous les cœurs du GPU
        d_next_grid[1:-1, 1:-1] = d_local_grid[1:-1, 1:-1] + self.alpha * (
            d_local_grid[0:-2, 1:-1] +  # Voisin du Haut
            d_local_grid[2:, 1:-1]   +  # Voisin du Bas
            d_local_grid[1:-1, 0:-2] +  # Voisin de Gauche
            d_local_grid[1:-1, 2:]   -  # Voisin de Droite
            4 * d_local_grid[1:-1, 1:-1]
        )
        return d_next_grid
```

### 7.3 Explication ligne par ligne

| Code | Ce que ça fait | Pourquoi c'est nécessaire |
|------|----------------|--------------------------|
| `import cupy as cp` | Importe CuPy avec l'alias `cp` (même convention que NumPy) | Permet d'utiliser des tableaux GPU avec une syntaxe quasi-identique à NumPy |
| `cp.copy(d_local_grid)` | Copie le tableau GPU dans un nouveau tableau | On ne modifie pas le tableau courant en place — cela corromprait le calcul des voisins |
| `d_local_grid[1:-1, 1:-1]` | Sélectionne toutes les cellules intérieures (sans les bords) | Les bords seront gérés par les halos — on les laisse à zéro ici |
| `d_local_grid[0:-2, 1:-1]` | Sélectionne la ligne du dessus pour chaque cellule | C'est le voisin "haut" du stencil |
| `d_local_grid[2:, 1:-1]` | Sélectionne la ligne du dessous | C'est le voisin "bas" du stencil |
| `self.alpha` | Coefficient de diffusion (0.1 par défaut) | Contrôle la vitesse de propagation de la chaleur |

> 💡 **Notation CuPy vs NumPy :** `d_` est une convention pour rappeler que la variable est sur le **D**evice (GPU). Les opérations sur des tableaux CuPy s'exécutent automatiquement sur le GPU — aucune syntaxe spéciale n'est nécessaire.

---

## 8. Étape 4 — La communication des frontières (MPI / Halos)

### 8.1 Concept : pourquoi des halos ?

Chaque processus calcule la diffusion sur sa bande. Mais les cellules situées sur la **dernière ligne** d'un processus ont besoin de connaître la **première ligne** du processus voisin du dessous pour calculer leur voisin "bas" — et vice versa.

Sans échange de halos, la simulation serait correcte à l'intérieur de chaque bande mais **incorrecte aux frontières** : les bords se comporteraient comme des murs isolants thermiques.

```
Processus 0 :          Processus 1 :
┌──────────┐           ┌──────────┐
│ ligne 38 │ ──────►   │ halo haut│  (copie de la ligne 38 du processus 0)
│ ligne 39 │           │ ligne 40 │
└──────────┘           │ ligne 41 │
                       │   ...    │
                       │ ligne 79 │
                       │ halo bas │ ──────► (envoyé au processus 2)
                       └──────────┘
```

### 8.2 Créer le fichier `halo_exchange.py`

```python
%%writefile src/halo_exchange.py
from mpi4py import MPI
import numpy as np

def exchange_halos(local_grid, mpi_mgmt):
    """
    Échange les lignes de frontières (halos) entre voisins.
    Ajoute une ligne virtuelle en haut et/ou en bas si nécessaire.
    """
    comm = mpi_mgmt.comm
    rank = mpi_mgmt.rank
    size = mpi_mgmt.size

    # Créer un tampon avec de l'espace pour les halos (lignes fantômes)
    # On ajoute une ligne en haut et une en bas
    height, width = local_grid.shape
    buffered_grid = np.zeros((height + 2, width), dtype=local_grid.dtype)
    buffered_grid[1:-1, :] = local_grid

    # Échange avec le voisin du HAUT
    # (envoi de notre première ligne, réception dans notre ligne 0)
    if mpi_mgmt.up_neighbor is not None:
        comm.Sendrecv(local_grid[0, :], dest=mpi_mgmt.up_neighbor, sendtag=11,
                      recvbuf=buffered_grid[0, :], source=mpi_mgmt.up_neighbor, recvtag=22)

    # Échange avec le voisin du BAS
    # (envoi de notre dernière ligne, réception dans notre dernière ligne fantôme)
    if mpi_mgmt.down_neighbor is not None:
        comm.Sendrecv(local_grid[-1, :], dest=mpi_mgmt.down_neighbor, sendtag=22,
                      recvbuf=buffered_grid[-1, :], source=mpi_mgmt.down_neighbor, recvtag=11)

    return buffered_grid
```

### 8.3 Explication ligne par ligne

| Code | Ce que ça fait | Pourquoi c'est nécessaire |
|------|----------------|--------------------------|
| `buffered_grid = np.zeros((height + 2, width))` | Crée un tableau avec 2 lignes supplémentaires (une en haut, une en bas) | Ces deux lignes "fantômes" accueilleront les halos reçus des voisins |
| `buffered_grid[1:-1, :] = local_grid` | Copie la grille locale dans la zone centrale du tampon | Les lignes 0 et -1 restent vides pour recevoir les halos |
| `comm.Sendrecv(...)` | Envoie et reçoit simultanément — évite les interblocages (*deadlocks*) | Si on utilisait `Send` puis `Recv` séparément, les deux processus pourraient se bloquer en attendant l'un l'autre |
| `sendtag=11` / `recvtag=22` | Étiquettes pour distinguer les messages "vers le haut" et "vers le bas" | Évite qu'un processus confonde un message de son voisin du haut avec celui du voisin du bas |
| `if mpi_mgmt.up_neighbor is not None` | Vérifie qu'il y a bien un voisin du haut | Le processus 0 n'a pas de voisin au-dessus — on ne doit pas envoyer de message dans le vide |

---

## 9. Étape 5 — Orchestration et lancement de la simulation

### 9.1 Créer le fichier `main_simulation.py`

Ce fichier est le chef d'orchestre. Il appelle les trois composants dans le bon ordre à chaque pas de temps.

```python
%%writefile src/main_simulation.py
import numpy as np
import cupy as cp
from mpi_manager import MPIManager
from halo_exchange import exchange_halos
from simulation_kernel import ClimateSimulationKernel

def main():
    # Configuration de la simulation (Grille globale de 120x120 pour le test)
    global_height, global_width = 120, 120
    steps = 10  # Nombre d'itérations de temps

    # 1. Initialisation de l'infrastructure MPI globale avec les dimensions requises
    mpi_mgmt = MPIManager(global_width, global_height)
    local_height = mpi_mgmt.local_height

    # 2. Initialisation des données locales sur le CPU (NumPy)
    np.random.seed(42)
    local_grid_cpu = np.zeros((local_height, global_width), dtype=np.float32)

    # On injecte de la chaleur sur le morceau du milieu (Rang 1)
    if mpi_mgmt.rank == 1:
        local_grid_cpu[2:8, 40:80] = 100.0

    # 3. Allocation et transfert vers le GPU local (CuPy)
    d_local_grid = cp.array(local_grid_cpu)

    # 4. Instanciation du Kernel GPU
    kernel = ClimateSimulationKernel(
        local_grid_shape=(local_height, global_width), alpha=0.1
    )

    if mpi_mgmt.rank == 0:
        print(f"[Master] Lancement de la simulation répartie sur {mpi_mgmt.size} processus...")

    # 5. Boucle principale de la simulation temporelle
    for step in range(steps):
        # Échange de halos (via CPU pour la communication MPI)
        h_grid = cp.asnumpy(d_local_grid)          # GPU → CPU
        buffered_h_grid = exchange_halos(h_grid, mpi_mgmt)

        # Calcul du stencil sur GPU
        d_buffered_grid = cp.array(buffered_h_grid)  # CPU → GPU
        d_next_buffered = kernel.compute_next_step(d_buffered_grid)

        # Extraction de la zone centrale utile (sans les lignes fantômes)
        d_local_grid = d_next_buffered[1:-1, :]

    # 6. Synchronisation finale
    mpi_mgmt.comm.Barrier()

    # 7. Collecte et reconstruction de la grille globale sur le Rang 0
    final_local_cpu = cp.asnumpy(d_local_grid)
    global_grid = mpi_mgmt.gather_grid(final_local_cpu)

    # 8. Affichage des résultats
    if mpi_mgmt.rank == 0:
        print(f"\n[Master] Collecte réussie ! Grille globale reconstruite : {global_grid.shape}")
        print(f"[Master] Température maximale détectée : {np.max(global_grid):.2f}°C")
        print(f"[Master] Température moyenne globale  : {np.mean(global_grid):.4f}°C")
        print("[Master] Fin du projet. Prêt pour l'analyse des résultats.")
    else:
        print(f"[Processus {mpi_mgmt.rank}] Données envoyées au Master et mémoire libérée.")

if __name__ == "__main__":
    main()
```

### 9.2 Comprendre la boucle principale

Voici ce qui se passe à chaque itération de temps :

```
┌─────────────────────────────────────────────────────┐
│ Pour chaque pas de temps (step) :                   │
│                                                     │
│  GPU → CPU : cp.asnumpy()                           │
│      ↓                                              │
│  Exchange halos (MPI communication réseau)          │
│      ↓                                              │
│  CPU → GPU : cp.array()                             │
│      ↓                                              │
│  Calcul stencil CuPy (sur GPU, en parallèle)        │
│      ↓                                              │
│  Extraction zone utile [1:-1, :]                    │
└─────────────────────────────────────────────────────┘
```

> ⚠️ **Transfert CPU↔GPU à chaque pas de temps :** C'est un compromis de ce projet. MPI ne peut communiquer qu'entre mémoires CPU. On doit donc rapatrier la grille du GPU vers le CPU pour l'échange de halos, puis la renvoyer sur le GPU pour le calcul. Dans des systèmes de production, on utiliserait **CUDA-Aware MPI** pour communiquer directement entre GPU, sans passer par le CPU.

### 9.3 Lancer la simulation

La commande `mpirun` lance plusieurs processus MPI en parallèle. L'option `-np 3` signifie "lance 3 processus".

```python
!PYTHONPATH=src mpirun --allow-run-as-root --oversubscribe -np 3 python3 src/main_simulation.py
```

**Décomposition de la commande :**

| Partie | Signification |
|--------|--------------|
| `PYTHONPATH=src` | Indique à Python de chercher les modules dans le dossier `src/` (sinon `import mpi_manager` échoue) |
| `mpirun` | Exécutable MPI qui lance les processus |
| `--allow-run-as-root` | Nécessaire sur Colab car l'utilisateur est `root` ; MPI refuse normalement d'être lancé en root pour des raisons de sécurité |
| `--oversubscribe` | Autorise plus de processus MPI que de cœurs CPU physiques disponibles (utile sur Colab qui n'a qu'un ou deux cœurs) |
| `-np 3` | Lance exactement 3 processus MPI parallèles |
| `python3 src/main_simulation.py` | Le script à exécuter sur chacun des 3 processus |

---

## 10. Résultats attendus

### 10.1 Sortie console normale

```
[Master] Lancement de la simulation répartie sur 3 processus...
[Processus 1] Données envoyées au Master et mémoire libérée.
[Processus 2] Données envoyées au Master et mémoire libérée.

[Master] Collecte réussie ! Grille globale reconstruite : (120, 120)
[Master] Température maximale détectée : 45.23°C
[Master] Température moyenne globale  : 1.2550°C
[Master] Fin du projet. Prêt pour l'analyse des résultats.
```

> 💡 **Les lignes des processus 1 et 2 peuvent apparaître avant ou après la ligne du Master** — c'est normal : l'ordre d'affichage des sorties distribuées n'est pas garanti.

### 10.2 Ce que valident ces résultats

| Indicateur | Ce qu'il prouve |
|------------|----------------|
| `Grille globale reconstruite : (120, 120)` | Le `gather_grid` a bien réassemblé les 3 bandes en une grille complète |
| `Température maximale` < 100°C | La chaleur initiale (100°C) s'est bien diffusée — physique correcte |
| `Température moyenne` > 0°C | Le calcul n'a pas dégénéré (zéros partout serait suspect) |
| Pas d'erreur MPI | Les échanges de halos ont fonctionné sans blocage ni corruption |

---

## 11. SOS / Dépannage

### 11.1 Erreurs d'installation

| Erreur | Cause probable | Solution |
|--------|----------------|----------|
| `ModuleNotFoundError: No module named 'mpi4py'` | Installation MPI échouée | Relancer `!pip install mpi4py` ; vérifier avec `!which mpirun` |
| `ModuleNotFoundError: No module named 'cupy'` | CuPy non installé ou mauvaise version CUDA | Vérifier la version CUDA avec `!nvcc --version` puis installer `cupy-cuda11x` ou `cupy-cuda12x` en conséquence |
| `ImportError: libcuda.so.1: cannot open shared object file` | CUDA mal configuré dans l'environnement | Redémarrer la session Colab et réinstaller les dépendances |
| `pip install cupy-cuda12x` très lent | Téléchargement de wheels CUDA volumineux | Patience — les packages CUDA font souvent plusieurs centaines de Mo |

### 11.2 Erreurs MPI

| Erreur | Cause probable | Solution |
|--------|----------------|----------|
| `[mpirun] CANNOT RUN AS ROOT` | Colab s'exécute en tant que root et MPI refuse | Ajouter `--allow-run-as-root` à la commande `mpirun` |
| `There are not enough slots available` | Plus de processus que de cœurs disponibles | Ajouter `--oversubscribe` à la commande `mpirun` |
| `[Errno 98] Address already in use` | Un processus MPI précédent tourne encore | Exécuter `!pkill -f python3` puis relancer |
| Processus bloqués indéfiniment | Interblocage MPI (*deadlock*) | Vérifier que `Sendrecv` est utilisé et non `Send`/`Recv` séparément |
| `MPI_ERR_RANK: invalid rank` | `up_neighbor` ou `down_neighbor` mal calculé | Vérifier que `None` est bien retourné pour les processus aux extrémités (rang 0 en haut, rang `size-1` en bas) |

### 11.3 Erreurs GPU / CuPy

| Erreur | Cause probable | Solution |
|--------|----------------|----------|
| `cupy.cuda.runtime.CUDARuntimeError: cudaErrorNoDevice` | Session Colab sans GPU | `Exécution → Modifier le type d'exécution → GPU T4` |
| `cupy.cuda.memory.OutOfMemoryError` | Grille trop grande pour la VRAM | Réduire `global_height` et `global_width` dans `main_simulation.py` |
| `AttributeError: 'numpy.ndarray' has no attribute '__cuda_array_interface__'` | Passage d'un tableau NumPy là où CuPy est attendu | Convertir avec `cp.array(mon_tableau_numpy)` avant le calcul GPU |
| Résultats incorrects (NaN ou Inf) | Coefficient `alpha` trop élevé → instabilité numérique | Réduire `alpha` (essayez `0.05` au lieu de `0.1`) |

### 11.4 Erreurs Python / imports

| Erreur | Cause probable | Solution |
|--------|----------------|----------|
| `ModuleNotFoundError: No module named 'mpi_manager'` | `PYTHONPATH` non défini | Ajouter `PYTHONPATH=src` avant la commande `mpirun` |
| `AttributeError: 'MPIManager' object has no attribute 'local_height'` | Ancienne version de `mpi_manager.py` | Réécrire le fichier avec `%%writefile src/mpi_manager.py` |
| `TypeError: __init__() takes 1 positional argument but 3 were given` | Appel à `MPIManager()` sans arguments | Passer les dimensions : `MPIManager(global_width, global_height)` |

### 11.5 Commandes de diagnostic rapide

```python
# Vérifier que MPI fonctionne
!mpirun --allow-run-as-root -np 3 python3 -c "from mpi4py import MPI; print('Rang', MPI.COMM_WORLD.Get_rank(), '/ Taille', MPI.COMM_WORLD.Get_size())"

# Vérifier que CuPy voit le GPU
import cupy as cp
print(cp.cuda.runtime.getDeviceProperties(0)['name'])

# Vérifier que les fichiers sources sont bien créés
!ls -la src/

# Afficher les processus Python en cours
!ps aux | grep python
```

---

## 12. Auteurs

**Équipe projet — Sujet 11 : Simulation Météo Distribuée avec GPU**

- **Malak Boussetta** — Scrum Master / Infrastructure MPI (`mpi_manager.py`, décomposition de domaine, orchestrateur principal)
- **Aya** — Développeuse GPU / CUDA Kernels (`simulation_kernel.py`, stencil de diffusion CuPy)
- **Marwa** — Ingénieure QA / Échange des Frontières (`halo_exchange.py`, communication MPI des halos)

Master d'Excellence en Intelligence Artificielle
Faculté des Sciences Ben M'Sik — Université Hassan II de Casablanca
Année universitaire 2025–2026
