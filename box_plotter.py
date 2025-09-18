from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import numpy as np

label_csv = Path("labels.csv")

def plot_boxplot(csv_path: Path) -> None:
    df = pd.read_csv(csv_path, delimiter=',')
    plt.figure(figsize=(12, 6))
    data = df.iloc[:, 1:].astype(int)
    names = df.iloc[:, 0].apply(lambda x: x.split('_')[-1])
    sns.boxplot(data=data.T)
    plt.xticks(ticks=range(len(names)), labels=names, rotation=45)
    plt.xlabel("Row Names")
    plt.ylabel("Label Values")
    plt.title("Boxplot of Label Distributions per Row")
    plt.tight_layout()
    plt.show()

def calculate_spearman_correlation(csv_path):
    pred_labels = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5])
    df = pd.read_csv(csv_path, delimiter=';')
    true_labels = df.iloc[:, 1:].mean(axis=1)
    rho, pval = spearmanr(pred_labels, true_labels)
    print(f"Spearman's rho = {rho:.3f}, p-value = {pval:.3g}")


if __name__ == "__main__":
    # plot_boxplot(label_csv)
    csv_file = Path("labels_merged.csv")
    calculate_spearman_correlation(csv_file)