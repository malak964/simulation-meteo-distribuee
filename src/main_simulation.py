import numpy as np
import cupy as cp
from mpi_manager import MPIManager
from halo_exchange import exchange_halos
from simulation_kernel import ClimateSimulationKernel

def main():
    # Configuration de la simulation (Grille globale de 120x120 pour le test)
    global_height, global_width = 120, 120
    steps = 10  # Nombre d'itérations de temps
    
    # 1. Initialisation de l'infrastructure MPI globale
    mpi_mgmt = MPIManager(global_width, global_height)
    local_height = mpi_mgmt.local_height
    
    # 2. Initialisation des données locales sur le CPU (NumPy)
    np.random.seed(42)
    local_grid_cpu = np.zeros((local_height, global_width), dtype=np.float32)
    
    # On injecte de la chaleur sur le morceau du milieu (Rank 1)
    if mpi_mgmt.rank == 1:
        local_grid_cpu[2:8, 40:80] = 100.0

    # 3. Allocation et transfert vers le GPU local (CuPy)
    d_local_grid = cp.array(local_grid_cpu)
    
    # 4. Instanciation du Kernel GPU d'Aya
    kernel = ClimateSimulationKernel(local_grid_shape=(local_height, global_width), alpha=0.1)
    
    if mpi_mgmt.rank == 0:
        print(f"[Master] Lancement de la simulation répartie sur {mpi_mgmt.size} processus...")

    # 5. Boucle principale de la simulation temporelle
    for step in range(steps):
        # Échange de halos (via CPU pour la communication MPI)
        h_grid = cp.asnumpy(d_local_grid)
        buffered_h_grid = exchange_halos(h_grid, mpi_mgmt)
        
        # Calcul du stencil sur GPU
        d_buffered_grid = cp.array(buffered_h_grid)
        d_next_buffered = kernel.compute_next_step(d_buffered_grid)
        
        # Extraction de la zone centrale utile
        d_local_grid = d_next_buffered[1:-1, :]
        
    # Synchronisation avant la collecte
    mpi_mgmt.comm.Barrier()
    
    # 6. COLLECTE ET CENTRALISATION DES DONNÉES
    # On rapatrie le résultat final du GPU vers le CPU avant le Gather MPI
    final_local_cpu = cp.asnumpy(d_local_grid)
    
    # Appel de ta méthode pour reconstruire la grille complète sur le Rank 0
    global_grid = mpi_mgmt.gather_grid(final_local_cpu)
    
    # 7. Validation et affichage du résultat final sur le Master
    if mpi_mgmt.rank == 0:
        print(f"\n[Master] Collecte réussie ! Grille globale reconstruite : {global_grid.shape}")
        print(f"[Master] Température maximale détectée : {np.max(global_grid):.2f}°C")
        print(f"[Master] Température moyenne globale : {np.mean(global_grid):.4f}°C")
        print("[Master] Fin du projet. Prêt pour l'analyse des résultats.")
    else:
        print(f"[Processus {mpi_mgmt.rank}] Données envoyées au Master et mémoire libérée.")

if __name__ == "__main__":
    main()
