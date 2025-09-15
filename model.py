import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from sklearn.decomposition import PCA as sklearnPCA
from initial_analysis import mean_image, PCA, extract_labels
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def forward_selection(model, X, y, X_test, y_test, pcs, cv = 5):
    scores = []
    for n_features in tqdm(range(1, pcs + 1), desc="Forward Selection"):
        sfs = SequentialFeatureSelector(model, n_features_to_select=n_features, direction='forward', cv=cv)
        sfs.fit(X, y)
        selected_features = sfs.get_support(indices=True)
        print(f"Selected features for {n_features} features: {selected_features}")
        X_selected = X[:, selected_features]
        model.fit(X_selected, y)
        test_selected = X_test[:, selected_features]
        score = model.score(test_selected, y_test)
        scores.append(score)
    
    plt.plot(range(1, pcs + 1), scores, marker='o')
    plt.xlabel('Number of Selected Features')
    plt.ylabel('R² Score on Test Set')
    plt.title('Forward Selection Performance')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    image_path = 'subset'
    mean_img = mean_image(image_path)
    pca_result = PCA(image_path, mean_img)
    labels = extract_labels(image_path)
    X_train, X_test, y_train, y_test = train_test_split(pca_result, labels, test_size=0.15, random_state=42)
    model = LinearRegression()
    forward_selection(model, X_train, y_train, X_test, y_test, pcs=50, cv = 5)