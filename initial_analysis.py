import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from sklearn.decomposition import PCA as sklearnPCA

def histogram_labels(image_path, bins = 3):
    labels = []
    for filename in os.listdir(image_path):
        if filename.endswith('.jpg'):
            parts = filename.split('_')
            label = parts[-1].split('.')[0]
            labels.append(int(label))
    plt.hist(labels, bins=bins, edgecolor='black')
    plt.xlabel('Labels')
    plt.title('Histogram of Labels')
    plt.xticks([1, 2, 3])
    plt.show()


def mean_image(image_path):
    images = []
    for filename in os.listdir(image_path):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(image_path, filename)).convert('L')
            img_array = np.array(img, dtype=np.float64) / 255.0  
            images.append(img_array)
    mean_img = np.mean(images, axis=0)
    # plt.imshow((mean_img * 255).astype(np.uint8))
    # plt.title('Mean Image')
    # plt.axis('off')
    # plt.show()
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
    explained_variance = pca.explained_variance_ratio_

    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.title('PCA Explained Variance')
    plt.xlim(1, 50)
    plt.show()

    components = pca.components_
    pc_1 = components[0]
    min_score = pca_result[:, 0].min()
    max_score = pca_result[:, 0].max()

    img_min = mean_flat + min_score * pc_1
    img_max = mean_flat + max_score * pc_1
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_min.reshape(mean_image.shape) * 255, cmap='gray')
    plt.title('Min weight added to PC1')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(mean_image, cmap='gray')
    plt.title('Mean Image')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(img_max.reshape(mean_image.shape) * 255, cmap='gray')
    plt.title('Max weight added to PC1')
    plt.axis('off')
    plt.show()





if __name__ == "__main__":
    image_path = 'subset'
    mean_img = mean_image(image_path)
    PCA(image_path, mean_img)
    # histogram_labels(image_path)
