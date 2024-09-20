import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import time
import os
import pandas as pd
from multiprocessing import Process, Value

def generate_points_in_circle(N, radius, seed=None):
    np.random.seed(seed)
    r = radius * np.sqrt(np.random.uniform(0, 1, N))
    theta = 2 * np.pi * np.random.uniform(0, 1, N)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points = np.vstack((x, y)).T
    return points

def apply_clustering(points, eps, min_samples, min_cluster_size, min_area):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    clusters = extract_cluster_properties(points, labels, min_cluster_size, min_area)
    return labels, clusters

def extract_cluster_properties(points, labels, min_cluster_size, min_area):
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
        clusters[label] = {
            'points': cluster_points,
            'N_prime': N_prime,
            'S_prime': S_prime,
        }
    return clusters

def compute_cluster_area(cluster_points):
    if len(cluster_points) < 3:
        return 0  # Area is zero if less than 3 points
    hull = ConvexHull(cluster_points)
    return hull.volume  # For 2D, 'volume' returns area

def save_simulation_data(data_file, S_prime, N_prime, iteration):
    df = pd.DataFrame({
        'S_prime': [S_prime],
        'N_prime': [N_prime],
        'iteration': [iteration]
    })
    if not os.path.exists(data_file):
        df.to_csv(data_file, index=False, mode='w')
    else:
        df.to_csv(data_file, index=False, mode='a', header=False)

def simulate_joint_distribution(N, radius, eps, min_samples, min_cluster_size, min_area, stop_flag, seed=None):
    iteration_times = []
    iteration = 0
    start_time = time.time()
    
    data_file = 'simulation_data.csv'
    if os.path.exists(data_file):
        print("Found existing simulation data.")
        joint_data_df = pd.read_csv(data_file)
        iteration = joint_data_df['iteration'].max()
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
                N_prime = cluster['N_prime']
                save_simulation_data(data_file, S_prime, N_prime, iteration)
            iter_end_time = time.time()
            iter_time = iter_end_time - iter_start_time
            iteration_times.append(iter_time)
            if len(iteration_times) > 1000:
                iteration_times.pop(0)
            if iteration % 1000 == 0:
                avg_time = np.mean(iteration_times)
                print(f"Iteration: {iteration}, Avg Time per Iteration (last 1000): {avg_time:.4f} seconds")
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        total_time = time.time() - start_time
        print(f"Total iterations completed: {iteration}")
        print(f"Total simulation time: {total_time:.2f} seconds")
        print("Data saved. You can resume the simulation later.")

def plotting_process(stop_flag, min_cluster_size, update_interval=5):
    data_file = 'simulation_data.csv'
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    last_iteration = 0
    while not stop_flag.value:
        if os.path.exists(data_file):
            joint_data_df = pd.read_csv(data_file)
            if not joint_data_df.empty:
                current_iteration = joint_data_df['iteration'].max()
                if current_iteration - last_iteration >= 1000:
                    last_iteration = current_iteration
                    S_values = joint_data_df['S_prime'].values
                    N_values = joint_data_df['N_prime'].values.astype(int)
                    min_N = min_cluster_size - 1
                    max_N = N_values.max() + 1
                    max_S = np.percentile(S_values, 99)
                    ax.clear()
                    bins_N = np.arange(min_N, max_N + 1, 1)
                    bins_S = np.linspace(0.5, max_S, 50)
                    hist, xedges, yedges = np.histogram2d(
                        S_values, N_values,
                        bins=[bins_S, bins_N],
                        density=False
                    )
                    hist /= (current_iteration / 1000)
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
    N = 10000
    radius = 100
    eps = 1.0
    min_samples = 10
    min_cluster_size = 10
    min_area = 0.5
    seed = None

    stop_flag = Value('i', 0)

    sim_process = Process(target=simulate_joint_distribution, args=(
        N, radius, eps, min_samples, min_cluster_size, min_area, stop_flag, seed
    ))

    plot_process = Process(target=plotting_process, args=(stop_flag, min_cluster_size))

    try:
        sim_process.start()
        plot_process.start()

        sim_process.join()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        stop_flag.value = 1
        plot_process.join()
        print("Processes terminated.")

if __name__ == "__main__":
    main()
