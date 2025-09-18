import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from sklearn.decomposition import PCA as sklearnPCA
from initial_analysis import mean_image, PCA
from label_data import label_data
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


def forward_selection(model, X_train, y_train, X_test, y_test, pcs, cv = 5):
    mse_scores = []
    avg_preds = []

    for n_features in tqdm(range(1, pcs + 1), desc="Forward Selection"):
        sfs = SequentialFeatureSelector(
            model, 
            n_features_to_select=n_features, 
            direction='forward', 
            cv=cv,
            scoring='neg_mean_squared_error')
        
        sfs.fit(X_train, y_train)
        selected_features = sfs.get_support(indices=True)
        print(f"Selected features for {n_features} features: {selected_features}")
        
        X_train_selected = X_train[:, selected_features]
        X_test_selected = X_test[:, selected_features]

        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)

        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)
        avg_preds.append(np.mean(y_pred))
        
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, pcs + 1), mse_scores, marker='o')
    plt.xlabel('Number of Selected PCs')
    plt.ylabel('Mean Squared Error on Test Set')
    plt.title('Forward Selection Performance')
    plt.savefig('forward_selection_performance.png')
    plt.grid()
    plt.show()
    # plt.figure(figsize=(8, 5))
    # plt.plot(range(1, pcs + 1), avg_preds, marker='o')
    # plt.xlabel('Number of PCs')
    # plt.ylabel('Average Predicted Rating')
    # plt.title('Average Predicted Rating vs Number of PCs')
    # plt.grid()
    # plt.show()

    return mse_scores


def plot_predicted_vs_features(model, X_test, selected_features):
    """
    Plots predicted ratings on the Y-axis vs. each selected feature on the X-axis.
    """
    X_test_selected = X_test[:, selected_features]
    y_pred = model.predict(X_test_selected)

    num_features = X_test_selected.shape[1]
    plt.figure(figsize=(5 * num_features, 4))

    for i in range(num_features):
        plt.subplot(1, num_features, i + 1)
        plt.scatter(X_test_selected[:, i], y_pred, alpha=0.6)
        plt.xlabel(f"PC {selected_features[i]+1}")
        plt.ylabel("Predicted Rating")
        plt.title(f"Predicted Rating vs PC {selected_features[i]+1}")
        plt.grid()

    plt.tight_layout()
    plt.show()

def plot_predicted_vs_features_colored(model, X_test, selected_features):
    """
    Single scatter plot:
    X-axis = feature values (all PCs stacked),
    Y-axis = predicted rating,
    Color = which PC the point came from.
    """
    X_test_selected = X_test[:, selected_features]
    y_pred = model.predict(X_test_selected)

    # Prepare data for stacked plot
    feature_values = []
    predicted_values = []
    feature_labels = []

    for i, pc_idx in enumerate(selected_features):
        feature_values.extend(X_test_selected[:, i])
        predicted_values.extend(y_pred)
        feature_labels.extend([f"PC {pc_idx+1}"] * X_test_selected.shape[0])

    feature_values = np.array(feature_values)
    predicted_values = np.array(predicted_values)
    predicted_values = predicted_values * 1.4 + 3
    feature_labels = np.array(feature_labels)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(feature_values, predicted_values,
                          c=[i for i in range(len(feature_labels))],
                          cmap='tab10', alpha=0.6)

    # Build legend
    unique_labels = np.unique(feature_labels)
    colors = [scatter.cmap(scatter.norm(i)) for i in range(len(unique_labels))]
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8)
               for c in colors]

    plt.xlabel("Feature Value (Of all Selected PCs)")
    plt.ylabel("Predicted Rating")
    plt.ylim(0, 5)
    plt.title("Predicted Rating vs. Feature Values (Colored by PC)")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    image_path = 'subset/subset/'
    csv_path = 'Labels_Cognitive.csv'
    mean_img = mean_image(image_path)
    pca, pca_result = PCA(image_path, mean_img)
    pca_result = np.concatenate([pca_result, pca_result], axis=0)
    labels = label_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(pca_result, labels, test_size=0.15, random_state=42)
    model = LinearRegression()
    forward_selection(model, X_train, y_train, X_test, y_test, pcs=10, cv = 5)

    # best_n = 5  # for example
    # sfs = SequentialFeatureSelector(model, n_features_to_select=best_n, direction='forward', cv=5)
    # sfs.fit(X_train, y_train)
    # selected_features = sfs.get_support(indices=True)

    # model.fit(X_train[:, selected_features], y_train)
    # # plot_predicted_vs_features(model, X_test, selected_features=selected_features)
    # plot_predicted_vs_features_colored(model, X_test, selected_features=selected_features)