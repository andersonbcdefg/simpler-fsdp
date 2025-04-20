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
    file_paths,
    labels,
    smooth_window=None,
    x_val = "step"
):
    if len(file_paths) != len(labels):
            raise ValueError("Number of file paths must match number of labels")

    for file_path, label in zip(file_paths, labels):
        try:
            if file_path.endswith(".txt"):
                # Read and convert data from file
                with open(file_path, 'r') as f:
                    losses = [float(x) for x in f.readlines()]
                times = None
                tokens = None
                if x_val != "step":
                    raise ValueError("Invalid x_val for .txt file")
            else:
                data = pd.read_json(file_path, orient="records", lines=True)
                print(data.columns)
                losses = data['loss'].to_list()
                first_timestamp = data['timestamp'].min()
                data['timestamp'] = (data['timestamp'] - first_timestamp).dt.total_seconds()
                times = data['timestamp'].to_list()
                tokens = data['tokens'].to_list()


            # Apply smoothing
            if smooth_window is not None:
                losses = smooth(losses, n=smooth_window)

            # Plot with label
            if x_val == "step":
                plt.plot(losses, label=label, alpha=0.5)
            elif x_val == "time":
                plt.plot(times, losses, label=label, alpha=0.3) # pyright: ignore
            elif x_val == "tokens":
                plt.plot(tokens, losses, label=label, alpha=0.3) # pyright: ignore
            else:
                raise ValueError(f"Invalid x_val: {x_val}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # Add legend, title and labels
    plt.legend()
    return plt
