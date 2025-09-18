import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from sklearn.decomposition import PCA as sklearnPCA

def mean_image(image_path):
    images = []
    for filename in os.listdir(image_path):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(image_path, filename)).convert('L')
            img_array = np.array(img, dtype=np.float64) / 255.0  
            images.append(img_array)
    mean_img = np.mean(images, axis=0)
    return mean_img

def PCA(image_path, mean_image):
    images = []
    mean_flat = mean_image.flatten()
    for filename in os.listdir(image_path):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(image_path, filename)).convert('L')
            img_array = (np.array(img, dtype=np.float64) / 255.0).flatten()
            img_array -= mean_flat
            images.append(img_array)
    pca = sklearnPCA()
    pca_result = pca.fit_transform(images)
    
    #FOR EXERCISE 3
    explained_variance = pca.explained_variance_ratio_
    # plt.figure(figsize=(10, 5))
    # plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center')
    # plt.xlabel('Principal Component')
    # plt.ylabel('Variance Explained')
    # plt.title('PCA Explained Variance')
    # plt.xlim(1, 50)
    # plt.show()

    #FOR EXERCISE 4
    components = pca.components_
    # components_to_analyze = [47, 67, 96, 259, 279]
    # fig, axes = plt.subplots(len(components_to_analyze), 3, figsize=(15, 15))
    # for row_idx, component in enumerate(components_to_analyze):
    #     pc_1 = components[component]
    #     min_score = pca_result[:, component].min()
    #     max_score = pca_result[:, component].max()

    #     img_min = mean_flat + min_score * pc_1
    #     img_max = mean_flat + max_score * pc_1

    #     axes[row_idx, 0].imshow(img_min.reshape(mean_image.shape) * 255, cmap='gray')
    #     axes[row_idx, 0].set_title(f'PC{component}: Min weight')
    #     axes[row_idx, 0].axis('off')

    #     axes[row_idx, 1].imshow(mean_image, cmap='gray')
    #     axes[row_idx, 1].set_title('Mean Image')
    #     axes[row_idx, 1].axis('off')

    #     axes[row_idx, 2].imshow(img_max.reshape(mean_image.shape) * 255, cmap='gray')
    #     axes[row_idx, 2].set_title(f'PC{component}: Max weight')
    #     axes[row_idx, 2].axis('off')

    # plt.tight_layout()
    # plt.show()
    return pca, pca_result





if __name__ == "__main__":
    image_path = 'subset/subset/'
    mean_img = mean_image(image_path)
    PCA(image_path, mean_img)
