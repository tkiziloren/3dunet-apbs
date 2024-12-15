from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def visualize_binding_site_prediction(grid_size=50, ligand_coords=None, radius=6.5):
    """
    Visualize the process of binding site prediction using a 3D grid.

    Parameters:
    - grid_size: The size of the 3D grid (assumed cubic for simplicity).
    - ligand_coords: List of ligand atom coordinates relative to the grid.
    - radius: Radius used to define the binding site around each ligand atom.
    """
    if ligand_coords is None:
        ligand_coords = [
            (25, 25, 25),  # Example ligand atom at the center
            (30, 25, 20),  # Nearby atom
            (20, 30, 30)  # Another atom
        ]

    # Create an empty 3D grid
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)

    # Mark binding site regions around each ligand atom
    for coord in ligand_coords:
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    if np.linalg.norm(np.array(coord) - np.array([x, y, z])) <= radius:
                        grid[x, y, z] = True

    # Extract binding site coordinates (True regions in the grid)
    binding_site_coords = np.argwhere(grid)

    # 3D Visualization
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot binding site
    ax.scatter(binding_site_coords[:, 0], binding_site_coords[:, 1], binding_site_coords[:, 2], c='blue', alpha=0.2, s=1)

    # Plot ligand atoms
    for coord in ligand_coords:
        ax.scatter(coord[0], coord[1], coord[2], c='red', s=50, label="Ligand Atom")

    ax.set_title("3D Visualization of Binding Site Prediction")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.legend(["Binding Site", "Ligand Atoms"])
    plt.show()


# Example ligand coordinates and visualization
example_ligand_coords = [(25, 25, 25), (30, 25, 20), (20, 30, 30)]
visualize_binding_site_prediction(ligand_coords=example_ligand_coords)