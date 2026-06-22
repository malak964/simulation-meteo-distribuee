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
    
    # 2. Décomposition de domaine : calcul de la taille du morceau pour ce processus
    local_height = mpi_mgmt.get_local_grid_shape(global_height)
    
    # 3. Initialisation des données locales sur le CPU (NumPy)
    np.random.seed(42)
    local_grid_cpu = np.zeros((local_height, global_width), dtype=np.float32)
    
    if mpi_mgmt.rank == 1:
        # On injecte de la chaleur artificielle sur le morceau du milieu pour voir la diffusion
        local_grid_cpu[2:8, 40:80] = 100.0

    # 4. Allocation et transfert des données vers le GPU local (CuPy)
    d_local_grid = cp.array(local_grid_cpu)
    
    # 5. Instanciation du Kernel GPU d'Aya
    kernel = ClimateSimulationKernel(local_grid_shape=(local_height, global_width), alpha=0.1)
    
    if mpi_mgmt.rank == 0:
        print(f"[Master] Lancement de la simulation répartie sur {mpi_mgmt.size} processus...")

    # 6. Boucle principale de la simulation temporelle
    for step in range(steps):
        # En provenance du code de Marwa : On repasse temporairement sur le CPU pour l'échange réseau MPI
        h_grid = cp.asnumpy(d_local_grid)
        buffered_h_grid = exchange_halos(h_grid, mpi_mgmt)
        
        # On renvoie la grille avec ses frontières à jour sur le GPU pour le calcul
        d_buffered_grid = cp.array(buffered_h_grid)
        
        # En provenance du code d'Aya : Calcul du stencil sur le GPU
        d_next_buffered = kernel.compute_next_step(d_buffered_grid)
        
        # On extrait la zone centrale utile (sans les lignes fantômes) pour l'étape suivante
        d_local_grid = d_next_buffered[1:-1, :]
        
    # Synchronisation finale
    mpi_mgmt.comm.Barrier()
    
    print(f"[Processus {mpi_mgmt.rank}] Simulation terminée avec succès. Grille locale calculée sur GPU.")

if __name__ == "__main__":
    main()
