import matplotlib
#matplotlib.use('TkAgg')  # Use 'TkAgg' backend for interactive plotting
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import time
import os
import pandas as pd
from multiprocessing import Process, Value

def generate_points_in_circle(N, radius, seed=None):
    """
    Generate N points uniformly distributed within a circle of given radius.
    """
    np.random.seed(seed)
    r = radius * np.sqrt(np.random.uniform(0, 1, N))
    theta = 2 * np.pi * np.random.uniform(0, 1, N)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points = np.vstack((x, y)).T
    return points

def apply_clustering(points, eps, min_samples, min_cluster_size, min_area):
    """
    Apply DBSCAN clustering algorithm to the given points.
    Returns cluster labels and properties for each cluster.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    clusters = extract_cluster_properties(points, labels, min_cluster_size, min_area)
    return labels, clusters

def extract_cluster_properties(points, labels, min_cluster_size, min_area):
    """
    Extract properties (S', lambda') for each cluster.
    Filters out clusters with fewer than min_cluster_size points or area less than min_area.
    """
    clusters = {}
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            continue  # Skip noise
        cluster_points = points[labels == label]
        N_prime = len(cluster_points)
        if N_prime < min_cluster_size:
            continue  # Skip small clusters
        S_prime = compute_cluster_area(cluster_points)
        if S_prime < min_area:
            continue  # Skip clusters with area less than min_area
        lambda_prime = N_prime / S_prime
        clusters[label] = {
            'points': cluster_points,
            'N_prime': N_prime,
            'S_prime': S_prime,
            'lambda_prime': lambda_prime
        }
    return clusters

def compute_cluster_area(cluster_points):
    """
    Compute the area of the cluster using Convex Hull.
    """
    if len(cluster_points) < 3:
        return 0  # Area is zero if less than 3 points
    hull = ConvexHull(cluster_points)
    return hull.volume  # For 2D, 'volume' returns area

def simulate_joint_distribution(N, radius, eps, min_samples, min_cluster_size, min_area, stop_flag, seed=None):
    """
    Run simulations to build the joint distribution of (S', lambda').
    """
    iteration_times = []
    iteration = 0
    start_time = time.time()

    # Load existing data if available
    data_file = 'simulation_data.csv'
    if os.path.exists(data_file):
        print("Found existing simulation data.")
        joint_data_df = pd.read_csv(data_file)
        iteration = joint_data_df['iteration'].max()
        iteration_times = joint_data_df.groupby('iteration')['iteration_time'].mean().tolist()
        print(f"Resuming from iteration {iteration + 1}.")
    else:
        print("No existing data found. Starting new simulation.")

    try:
        while not stop_flag.value:
            iteration += 1
            iter_start_time = time.time()
            sim_seed = np.random.randint(0, 1e9)
            points = generate_points_in_circle(N, radius, seed=sim_seed)
            labels, clusters = apply_clustering(points, eps, min_samples, min_cluster_size, min_area)
            for cluster in clusters.values():
                S_prime = cluster['S_prime']
                lambda_prime = cluster['lambda_prime']
                # Save data after each cluster
                save_simulation_data(data_file, S_prime, lambda_prime, iteration, time.time() - iter_start_time)
            iter_end_time = time.time()
            iteration_times.append(iter_end_time - iter_start_time)

            # Print metrics every 1000 iterations
            if iteration % 1000 == 0:
                avg_time = np.mean(iteration_times)
                print(f"Iteration: {iteration}, Avg Time per Iteration: {avg_time:.4f} seconds")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        total_time = time.time() - start_time
        print(f"Total iterations completed: {iteration}")
        print(f"Total simulation time: {total_time:.2f} seconds")
        print("Data saved. You can resume the simulation later.")

def save_simulation_data(data_file, S_prime, lambda_prime, iteration, iteration_time):
    """
    Save simulation data to a CSV file.
    """
    df = pd.DataFrame({
        'S_prime': [S_prime],
        'lambda_prime': [lambda_prime],
        'iteration': [iteration],
        'iteration_time': [iteration_time]
    })
    if not os.path.exists(data_file):
        df.to_csv(data_file, index=False, mode='w')
    else:
        df.to_csv(data_file, index=False, mode='a', header=False)

def plotting_process(stop_flag, update_interval=5):
    """
    Process that reads the saved data and updates the histogram periodically.
    """
    data_file = 'simulation_data.csv'
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(8, 6))
    last_iteration = 0
    while not stop_flag.value:
        if os.path.exists(data_file):
            joint_data_df = pd.read_csv(data_file)
            if not joint_data_df.empty:
                current_iteration = joint_data_df['iteration'].max()
                # Update plot every 1000 iterations
                if current_iteration - last_iteration >= 1000:
                    last_iteration = current_iteration
                    S_values = joint_data_df['S_prime'].values
                    lambda_values = joint_data_df['lambda_prime'].values
                    N_values = lambda_values * S_values  # Calculate N' from existing data

                    # Limit axes ranges to focus on relevant data
                    max_S = np.percentile(S_values, 99)
                    max_N = np.percentile(N_values, 99)

                    ax.clear()
                    # Compute the 2D histogram for S' vs. N'
                    hist, xedges, yedges = np.histogram2d(
                        S_values, N_values,
                        bins=50,
                        range=[[0.5, max_S], [min(N_values), max_N]],
                        density=False
                    )
                    # Normalize frequency per 1000 iterations
                    hist /= (current_iteration / 1000)
                    # Plot the normalized histogram
                    X, Y = np.meshgrid(xedges, yedges)
                    pcm = ax.pcolormesh(X, Y, hist.T, cmap='viridis')
                    ax.set_xlabel("Cluster Area (S')")
                    ax.set_ylabel("Number of Points in Cluster (N')")
                    ax.set_title("Real-Time Distribution of Cluster Area vs. Number of Points")
                    cbar = plt.colorbar(pcm, ax=ax, label='Average Frequency per 1000 Iterations')
                    plt.tight_layout()
                    plt.draw()
                    plt.pause(0.01)
                    cbar.remove()
        time.sleep(update_interval)
    plt.ioff()
    plt.show()

def main():
    """
    Main function to orchestrate the updated steps.
    """
    # Parameters
    N = 10000  # Number of points
    radius = 100  # Radius of the circular area
    eps = 1.0  # DBSCAN epsilon parameter (adjust based on scale)
    min_samples = 10  # Increased to prevent small clusters
    min_cluster_size = 10  # Minimum number of points in a cluster
    min_area = 0.5  # Minimum cluster area
    seed = None  # Random seed for reproducibility

    # Flag to signal processes to stop
    stop_flag = Value('i', 0)  # Shared integer (0 = False, 1 = True)

    # Start the simulation process
    sim_process = Process(target=simulate_joint_distribution, args=(
        N, radius, eps, min_samples, min_cluster_size, min_area, stop_flag, seed
    ))

    # Start the plotting process
    plot_process = Process(target=plotting_process, args=(stop_flag,))

    try:
        sim_process.start()
        plot_process.start()

        # Wait for the simulation process to finish
        sim_process.join()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        # Signal the plotting process to stop
        stop_flag.value = 1
        plot_process.join()
        print("Processes terminated.")

if __name__ == "__main__":
    main()
