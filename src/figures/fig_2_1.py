import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

# Set random seed for reproducibility
np.random.seed(42)

def generate_data(N=100):
    # Generate 10 means for each class
    means_blue = np.random.multivariate_normal([1, 0], np.eye(2), 10)
    means_orange = np.random.multivariate_normal([0, 1], np.eye(2), 10)

    # Generate observations
    X_blue = []
    for _ in range(N):
        m_idx = np.random.randint(0, 10)
        m = means_blue[m_idx]
        X_blue.append(np.random.multivariate_normal(m, np.eye(2)/5))

    X_orange = []
    for _ in range(N):
        m_idx = np.random.randint(0, 10)
        m = means_orange[m_idx]
        X_orange.append(np.random.multivariate_normal(m, np.eye(2)/5))

    X_blue = np.array(X_blue)
    X_orange = np.array(X_orange)

    X = np.vstack([X_blue, X_orange])
    y = np.array([0]*N + [1]*N) # 0 for Blue, 1 for Orange
    return X, y

def plot_decision_boundary(ax, model, X, y, title):
    # Create meshgrid
    h = 0.05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict
    mesh_points = np.c_[xx.ravel(), yy.ravel()]

    if isinstance(model, LinearRegression):
        Z = model.predict(mesh_points)
        Z = (Z > 0.5).astype(int) # Cutoff at 0.5 for regression
    else:
        Z = model.predict(mesh_points)

    Z = Z.reshape(xx.shape)

    # Plot contours
    cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA'])

    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.3)
    ax.contour(xx, yy, Z, colors='k', linewidths=0.5)

    # Plot training points
    ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='o', s=20, edgecolors='k', label='Blue Class')
    ax.scatter(X[y==1, 0], X[y==1, 1], c='orange', marker='o', s=20, edgecolors='k', label='Orange Class')

    ax.set_title(title)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

def main():
    X, y = generate_data(100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X, y)
    plot_decision_boundary(axes[0], lr, X, y, "Linear Regression")

    # 15-Nearest Neighbors
    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X, y)
    plot_decision_boundary(axes[1], knn, X, y, "15-Nearest Neighbors")

    plt.tight_layout()
    output_path = 'src/figures/output/fig_2_1.png'
    plt.savefig(output_path, dpi=300)
    print(f"Figure 2.1 generated in {output_path}")

if __name__ == "__main__":
    main()
