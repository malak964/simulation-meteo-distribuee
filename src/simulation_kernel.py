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
