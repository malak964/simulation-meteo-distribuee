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
