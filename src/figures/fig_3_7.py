import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define basis vectors for the subspace (plane)
    # Let's make them non-orthogonal to be general
    x1 = np.array([1, 0.5, 0])
    x2 = np.array([0, 1, 0.5])

    # Normalize for better visualization length
    x1 = x1 / np.linalg.norm(x1)
    x2 = x2 / np.linalg.norm(x2)

    # Create a grid for the plane
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 10), np.linspace(-0.5, 1.5, 10))

    # The plane is spanned by x1 and x2.
    # Any point p = a*x1 + b*x2
    # We need the normal to the plane to plot it easily using ax.plot_surface
    normal = np.cross(x1, x2)
    # Plane equation: ax + by + cz = 0 (passing through origin)
    # z = (-ax - by) / c
    z_plane = (-normal[0] * xx - normal[1] * yy) / normal[2]

    # Plot the plane (Subspace spanned by X)
    ax.plot_surface(xx, yy, z_plane, alpha=0.2, color='gray')

    # Define y vector (outside the plane)
    y_vec = np.array([0.8, 0.8, 0.8])

    # Calculate projection y_hat onto the plane spanned by x1, x2
    X = np.column_stack((x1, x2))
    # beta = (X^T X)^-1 X^T y
    beta = np.linalg.inv(X.T @ X) @ X.T @ y_vec
    y_hat = X @ beta

    # Residual
    residual = y_vec - y_hat

    # Plot vectors
    # Origin
    O = np.array([0, 0, 0])

    # Helper to plot vector
    def plot_vector(v, color, label, linestyle='-'):
        ax.quiver(0, 0, 0, v[0], v[1], v[2], color=color, arrow_length_ratio=0.1, linestyle=linestyle)
        ax.text(v[0], v[1], v[2], label, fontsize=12, color=color)

    # Plot x1, x2 (basis)
    plot_vector(x1, 'blue', r'$x_1$')
    plot_vector(x2, 'blue', r'$x_2$')

    # Plot y
    plot_vector(y_vec, 'red', r'$y$')

    # Plot y_hat
    plot_vector(y_hat, 'green', r'$\hat{y}$')

    # Plot residual (dashed line from y_hat to y)
    ax.plot([y_hat[0], y_vec[0]], [y_hat[1], y_vec[1]], [y_hat[2], y_vec[2]], 'k--', label=r'$y - \hat{y}$')
    ax.text((y_hat[0]+y_vec[0])/2, (y_hat[1]+y_vec[1])/2, (y_hat[2]+y_vec[2])/2, r'$\epsilon$', fontsize=12)

    # Set limits
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 1.5])
    ax.set_zlim([-0.5, 1.5])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_title('Geometric Interpretation of Least Squares (Figure 3.7)')

    # Adjust view
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    output_path = 'src/figures/output/fig_3_7.png'
    plt.savefig(output_path, dpi=300)
    print(f"Figure 3.7 generated in {output_path}")

if __name__ == "__main__":
    main()
