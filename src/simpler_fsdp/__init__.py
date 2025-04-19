import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def smooth(arr, n=20):
    """
    Smooths a 1D array using a sliding window average.

    Args:
        arr: Input array to smooth
        n: Half window size (full window is 2n+1)

    Returns:
        Smoothed array
    """
    # Convert to float to handle integer arrays properly
    arr = np.asarray(arr, dtype=float)

    # Create padded array to handle boundaries
    padded = np.pad(arr, (n, n), mode='edge')

    # Use cumulative sum for efficient windowed averaging
    cumsum = np.cumsum(padded)

    # Calculate sliding window by taking difference of cumulative sums
    # and dividing by window size
    window_sums = cumsum[2*n:] - cumsum[:-2*n]
    window_size = 2*n + 1

    return window_sums / window_size

def plot_multiple_files(
    file_paths, labels, smooth_window=20
):
    """
    Plot data from multiple files with smoothing.

    Args:
        file_paths: List of file paths containing the data (one value per line)
        labels: List of labels for each file's data
        smooth_window: Window size for smoothing (half window, default=20)
        title: Plot title (optional)
        xlabel: X-axis label (optional)
        ylabel: Y-axis label (optional)
        figsize: Figure size as tuple (width, height)
        save_path: Path to save the figure (optional)

    Returns:
        The figure and axes objects
    """
    if len(file_paths) != len(labels):
        raise ValueError("Number of file paths must match number of labels")

    for file_path, label in zip(file_paths, labels):
        try:
            if file_path.endswith(".txt"):
                # Read and convert data from file
                with open(file_path, 'r') as f:
                    data = [float(x) for x in f.readlines()]
            else:
                data = pd.read_json(file_path, orient="records", lines=True)['loss'].to_list()


            # Apply smoothing
            smoothed_data = smooth(data, n=smooth_window)

            # Plot with label
            plt.plot(smoothed_data, label=label)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # Add legend, title and labels
    plt.legend()
    return plt
