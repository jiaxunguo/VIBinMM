import numpy as np
import matplotlib.pyplot as plt

def plot_sphere_data(X, labels, title):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    # Make data
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xframe = 1 * np.outer(np.cos(u), np.sin(v))
    yframe = 1 * np.outer(np.sin(u), np.sin(v))
    zframe = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

    # Revalue the labels from big to small.
    unique_classes, counts = np.unique(labels, return_counts=True)
    unique_classes = unique_classes[np.argsort(counts)]

    color_map = plt.get_cmap('tab10')

    Z = [r"$\omega_1=-40, \omega_2=0$", r"$\omega_1=-40, \omega_2=-20$", r"$\omega_1=-40, \omega_2=-40$"]

    for label in unique_classes:
        mask = np.where(labels == label)

        _x = X[mask, 0]
        _y = X[mask, 1]
        _z = X[mask, 2]

        color = color_map(np.where(unique_classes == label)[0])

        ax.scatter(_x, _y, _z, c=color, s=10, label='Class '+str(label+1))

    # Plot the surface
    ax.plot_wireframe(xframe, yframe, zframe, alpha=0.1, linewidth=0.5)

    # Set an equal aspect ratio
    ax.set_aspect('equal')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # ax.invert_xaxis()

    ax.legend(fontsize=20)

    plt.title(title, fontsize=20)

    plt.show()