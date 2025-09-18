from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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




if __name__ == "__main__":
    plot_boxplot(label_csv)