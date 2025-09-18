from initial_analysis import mean_image, PCA
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA as sklearnPCA
import numpy as np
import matplotlib.pyplot as plt
from label_data import label_data


def generate_pc_vector_for_rating(model, X_train_selected, target_rating):
    coefs = model.coef_   
    intercept = model.intercept_

    mean_pc_vector = np.mean(X_train_selected, axis=0)

    current_rating = intercept + np.dot(mean_pc_vector, coefs)

    direction = coefs / np.linalg.norm(coefs)
    step = np.dot(direction, coefs)
    t = (target_rating - current_rating) / step

    
    new_pc_vector = mean_pc_vector + t * direction
    return new_pc_vector

def reconstruct_image_from_pcs(pca, pc_vector, image_shape, mean_image):

    reconstructed_image_uint8 = (np.clip(
        pca.inverse_transform(pc_vector.reshape(1, -1)).reshape(image_shape) + mean_image, 0, 1
    ) * 255).astype(np.uint8)
    return reconstructed_image_uint8


if __name__ == "__main__":
    image_path = 'subset'
    csv_path = 'Labels_Cognitive.csv'
    mean_img = mean_image(image_path)
    pca, pca_result = PCA(image_path, mean_img)
    pca_result = np.concatenate([pca_result, pca_result], axis=0)
    labels = label_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(pca_result, labels, test_size=0.15, random_state=42)
    
    model = LinearRegression()

    best_n = 5
    sfs = SequentialFeatureSelector(model, n_features_to_select=best_n, direction='forward', cv=5)
    sfs.fit(X_train, y_train)
    selected_features = sfs.get_support(indices=True)
    model.fit(X_train[:, selected_features], y_train)

    X_train_selected = X_train[:, selected_features]
    target_rating = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]
    fig, axes = plt.subplots(1, len(target_rating), figsize=(20, 5))
    for i, rating in enumerate(target_rating):
        new_pc_vector = generate_pc_vector_for_rating(model, X_train_selected, rating)
        full_pc_vector = np.zeros(pca.n_components_)
        full_pc_vector[selected_features] = new_pc_vector
        synthetic_image = reconstruct_image_from_pcs(
            pca=pca, mean_image=mean_img, pc_vector=full_pc_vector, image_shape=mean_img.shape)
        axes[i].imshow(synthetic_image, cmap='gray')
        axes[i].set_title(f"Rating {rating}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()