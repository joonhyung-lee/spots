import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def get_zero_map(resolution=(5,5,5)):
    """
        Get the zero map: (N x 3) array
    """
    zero_map = np.zeros((resolution[0]*resolution[1]*resolution[2], 3))

    return zero_map

def get_score_map(pcd, nbins=30, bandwidth=0.1, PLOT=False, weights=None):
    """
        Get the score of the physical safety score
    """
    x, y, z = pcd[:, 0], pcd[:, 1], pcd[:, 2]

    # Create KDE
    if weights is None:
        pass
    else:
        weights = weights.squeeze()
    # weights = np.ones((pcd.shape[0])) if weights is None else weights
    density = stats.gaussian_kde(pcd.T, bw_method=bandwidth, weights=weights)
    nbins = nbins
    xi, yi, zi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j, z.min():z.max():nbins*1j]

    # Compute density
    coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
    di = density(coords).reshape(xi.shape)

    # Normalize density (around 1)
    di_normalized = di / np.max(di)
    # score_physical = di_normalized

    if PLOT:
        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(xi.flatten(), yi.flatten(), zi.flatten(), c=di_normalized.flatten(), cmap='viridis', edgecolor='0.1', alpha=0.3)

        plt.xlabel('X')
        plt.ylabel('Y')
        ax.set_zlabel('Z')
        plt.colorbar(scatter, ax=ax)
        plt.show()

    return di_normalized, di

def get_score_map_penalty(pcd, nbins=30, bandwidth=0.1, penalty=0.5, PLOT=False):
    """
        Get the score of the physical safety score
    """
    x, y, z = pcd[:, 0], pcd[:, 1], pcd[:, 2]

    # Create KDE
    density = stats.gaussian_kde(pcd.T, bw_method=bandwidth)
    nbins = nbins
    xi, yi, zi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j, z.min():z.max():nbins*1j]

    # Compute density
    coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
    di = density(coords).reshape(xi.shape)

    # Normalize density (around 1)
    di_normalized = di / np.max(di)
    score_physical = di_normalized

    # Get the penalty score
    # score_penalty = np.exp(-penalty * di_normalized)
    score_penalty = 1 - penalty * di_normalized
    
    # Get the final score
    score_final = score_physical - score_penalty
    # score_final[score_final < 0] = 0

    if PLOT:
        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(xi.flatten(), yi.flatten(), zi.flatten(), c=score_final.flatten(), cmap='viridis', edgecolor='0.1', alpha=0.3)

        plt.xlabel('X')
        plt.ylabel('Y')
        ax.set_zlabel('Z')
        plt.colorbar(scatter, ax=ax)
        plt.show()

    return score_final, di

def sample_pcd_from_score_map(score_map, pcd, di, nbins, num_samples=10):
    """
        Sample the pcd from the score_map
    """
    # Flatten the normalized score_physical
    score_map_flat = score_map.flatten()
    score_map_norm = score_map_flat / np.sum(score_map_flat)

    sampled_indices = np.random.choice(len(score_map_flat), size=num_samples, p=score_map_norm)
    sampled_coords = np.array(np.unravel_index(sampled_indices, di.shape)).T

    # Convert indices to actual coordinates
    x, y, z = pcd[:, 0], pcd[:, 1], pcd[:, 2]
    x_range = np.linspace(x.min(), x.max(), nbins)
    y_range = np.linspace(y.min(), y.max(), nbins)
    z_range = np.linspace(z.min(), z.max(), nbins)

    sampled_xyz = np.array([x_range[sampled_coords[:, 0]], y_range[sampled_coords[:, 1]], z_range[sampled_coords[:, 2]]]).T

    return sampled_xyz

def plot_score_map(score_map, pcd, nbins):
    """
        Plot the score map
    """
    x, y, z = pcd[:, 0], pcd[:, 1], pcd[:, 2]
    xi, yi, zi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j, z.min():z.max():nbins*1j]

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    origin_pcd = ax.scatter(x, y, z, c='r', marker='o', s=10)

    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter(x, y, z, c='r', marker='o')
    ax.scatter(xi.flatten(), yi.flatten(), zi.flatten(), c=score_map.flatten(), cmap='viridis', alpha=0.01)
    plt.colorbar(origin_pcd, ax=ax)
    plt.show()

def plot_score_map_w_pcd(score_map, pcd, nbins):
    """
        Plot the score map
    """
    x, y, z = pcd[:, 0], pcd[:, 1], pcd[:, 2]
    xi, yi, zi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j, z.min():z.max():nbins*1j]

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    origin_pcd = ax.scatter(x, y, z, c='r', marker='o', s=10)

    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter(x, y, z, c='r', marker='o')
    ax.scatter(xi.flatten(), yi.flatten(), zi.flatten(), c=score_map.flatten(), cmap='viridis', alpha=0.01)
    plt.colorbar(origin_pcd, ax=ax)
    plt.show()