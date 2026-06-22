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
    
    # Échange avec le voisin du HAUT (envoi de notre première ligne, réception dans notre ligne 0)
    if mpi_mgmt.up_neighbor is not None:
        comm.Sendrecv(local_grid[0, :], dest=mpi_mgmt.up_neighbor, sendtag=11,
                      recvbuf=buffered_grid[0, :], source=mpi_mgmt.up_neighbor, recvtag=22)
                      
    # Échange avec le voisin du BAS (envoi de notre dernière ligne, réception dans notre dernière ligne)
    if mpi_mgmt.down_neighbor is not None:
        comm.Sendrecv(local_grid[-1, :], dest=mpi_mgmt.down_neighbor, sendtag=22,
                      recvbuf=buffered_grid[-1, :], source=mpi_mgmt.down_neighbor, recvtag=11)
                      
    return buffered_grid
