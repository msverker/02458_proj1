from pathlib import Path
from PIL import Image
import os
import pandas as pd
import numpy as np


def label_data(csv_path):
    df = pd.read_csv(csv_path, delimiter=';')
    richard_labels = df.iloc[:, -2].str.replace(',', '.').astype('float16')[:-2]
    mads_labels = df.iloc[:, -1].str.replace(',', '.').astype('float16')[:-2]
    assembled_labels = pd.concat([richard_labels, mads_labels], axis=0).to_numpy()
    return assembled_labels
        

if __name__ == "__main__":
    csv_file_path = 'Labels_Cognitive.csv'
    labels = label_data(csv_file_path)