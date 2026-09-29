import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# we choose as bin size the square root of the number of objects in our database sqrt(2483) ~= 50
N_BIN = 50


def plot_distribution(df, column, label):
    data = df[column]
    mean_v = data.mean()
    median_v = data.median()
    print(f"{label} mean: {mean_v}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # left: linear bins, linear scale
    axes[0].hist(data, bins=N_BIN)
    axes[0].set_title(f"Distribution of {label} counts (linear)")
    axes[0].set_xlabel(f"Number of {label}")

    # right: log-spaced bins on a log x-axis - reveals structure the linear
    # view hides, since the raw counts are heavily right-skewed
    log_bins = np.logspace(np.log10(data.min()), np.log10(data.max()), N_BIN + 1)
    axes[1].hist(data, bins=log_bins)
    axes[1].set_xscale("log")
    axes[1].set_title(f"Distribution of {label} counts (log-spaced bins)")
    axes[1].set_xlabel(f"Number of {label} (log scale)")

    for ax in axes:
        ax.axvline(mean_v, color='red', linestyle='dashed', linewidth=2, label=f'Mean = {mean_v:.1f}')
        ax.axvline(median_v, color='green', linestyle='dotted', linewidth=2, label=f'Median = {median_v:.1f}')
        ax.set_ylabel("Number of shapes")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()


def plot_stats(df):
    plot_distribution(df, 'num_vertices', 'vertices')
    plot_distribution(df, 'num_faces', 'faces')

    plt.figure(figsize=(16, 6))
    df['class'].value_counts().plot(kind='bar', width=0.85)
    plt.xlabel("Class")
    plt.ylabel("Number of shapes")
    plt.title("Number of Shapes per Class")
    plt.xticks(fontsize=7, rotation=90)
    plt.tight_layout()

    plt.show()