# Main PCA Digitos


# Imports
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Machine Learning Imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Imports Files
from func import *

if __name__ == "__main__":
    dataset_size = 60000
    dataset_size_test = 10000

    # Read Dataset
    dataset_images_train = read_dataset_images_train(dataset_size)
    dataset_labels_train = read_dataset_labels_train(dataset_size)
    dataset_images_test = read_dataset_images_test(dataset_size_test)
    dataset_labels_test = read_dataset_labels_test(dataset_size_test)

    # Flatten Dataset
    dataset_images_train_flat = dataset_images_train.reshape(
        dataset_images_train.shape[0], -1
    )
    dataset_images_test_flat = dataset_images_test.reshape(
        dataset_images_test.shape[0], -1
    )

    print("")
    print("Dataset Train Images: ", dataset_images_train.shape)
    print("Dataset Test Images: ", dataset_images_test.shape)
    print("Dataset Train Labels: ", dataset_labels_train.shape)
    print("Dataset Test Labels: ", dataset_labels_test.shape)

    print("")
    print("Dataset Train Images Flat: ", dataset_images_train_flat.shape)
    print("Dataset Test Images Flat: ", dataset_images_test_flat.shape)
    
    # Mean of the images from the dataset - Train
    print("")
    data_mean_images = average_number(dataset_images_train, dataset_labels_train)
    data_mean_image = np.mean(data_mean_images, axis=0)
    data_mean_image_flat = data_mean_image.flatten()

    print("Data Train Mean Images: ", data_mean_images.shape)
    print("Data Train Mean Image: ", data_mean_image.shape)
    print("Data Train Mean Image flat: ", data_mean_image_flat.shape)

    # Mean of the images from the dataset - Test
    print("")
    data_mean_images_test = average_number(dataset_images_test, dataset_labels_test)
    data_mean_image_test = np.mean(data_mean_images_test, axis=0)
    data_mean_image_test_flat = data_mean_image_test.flatten()

    print("Data Test Mean Images: ", data_mean_images_test.shape)
    print("Data Test Mean Image: ", data_mean_image_test.shape)
    print("Data Test Mean Image Flat: ", data_mean_image_test_flat.shape)

    # Center and normalize images
    data_images_center_train = (dataset_images_train_flat - data_mean_image_flat) / 255
    data_images_center_test = (
        dataset_images_test_flat - data_mean_image_test_flat
    ) / 255

    print("")
    print("Data Train Center Image: ", data_images_center_train.shape)
    print("Data Test Center Image: ", data_images_center_test.shape)
    
    # Scatter Matrix
    #scatter_matrix = scatter_matrix(data_images_center_train)
    #print("Scatter Matrix: ", scatter_matrix.shape)

    # Conv Matrix
    cov_matrix = np.cov(data_images_center_train, rowvar=False)
    print("Cov Matrix: ", cov_matrix.shape)

    # Eigenvalues and Eigenvectors
    # Não sei porque está a tipar para complexo 0j.
    
    # Scatter Matrix
    #eigenvalues_sc, eigenvectors_sc = np.linalg.eig(scatter_matrix)
    #eigenvalues_sc = eigenvalues_sc.astype("float32")
    #eigenvectors_sc = eigenvectors_sc.astype("float32")
    
    # Covariance Matrix
    eigenvalues_cov, eigenvectors_cov = np.linalg.eig(cov_matrix)
    eigenvalues_cov = eigenvalues_cov.astype("float32")
    eigenvectors_cov = eigenvectors_cov.astype("float32")

    print("")
    #print("Eigenvalues Scatter: ", eigenvalues_sc.shape)
    #print("Eigenvectors Scatter: ", eigenvectors_sc.shape)
    print("Eigenvalues Cov: ", eigenvalues_cov.shape)
    print("Eigenvectors Cov: ", eigenvectors_cov.shape)

    # Sort Eigenvalues - Eigenvectors in descending order
    #sorted_ind_sc = np.argsort(eigenvalues_sc)[::-1]
    #eigenvalues_sc = eigenvalues_sc[sorted_ind_sc]
    #eigenvectors_sc = eigenvectors_sc[:, sorted_ind_sc]

    sorted_ind_cov = np.argsort(eigenvalues_cov)[::-1]
    eigenvalues_cov = eigenvalues_cov[sorted_ind_cov]
    eigenvectors_cov = eigenvectors_cov[:, sorted_ind_cov]

    #print("Sorted Index SC: ", sorted_ind_sc)
    #print("Sorted Index Cov: ", sorted_ind_cov)

    # Covariance and Scatter Matrix Trace
    cov_trace = np.trace(cov_matrix)
    sum_eigenvalues_cov = sum(eigenvalues_cov)
    print("")
    print("Cov Trace: ", cov_trace)
    print("Sum Eigenvalues Cov: ", sum_eigenvalues_cov)

    # Eigenvectors Weight
    info = 0.90
    sum_info = 0
    number_eigenvectors = 0
    list_eigenvectors_weight = []

    for value in eigenvalues_cov:
        info_vector = value / cov_trace
        list_eigenvectors_weight.append(info_vector)

        if sum_info < info:
            sum_info += info_vector
            number_eigenvectors += 1

    print("")
    print("Number of Eigenvectors: ", number_eigenvectors)

    top_eigenvectors = eigenvectors_cov[:, :number_eigenvectors]
    print("")
    print("Top Eignvectors: ", top_eigenvectors.shape)

    # Project the dataset onto the Eigenvectors
    dataset_proj_train = np.dot(data_images_center_train, top_eigenvectors)
    dataset_proj_test = np.dot(data_images_center_test, top_eigenvectors)
    print("")
    print("Dataset Proj Train: ", dataset_proj_train.shape)
    print("Dataset Proj Test: ", dataset_proj_test.shape)

    # ToDo! -> Classifier
    knc = KNeighborsClassifier(n_neighbors=50)
    knc.fit(dataset_proj_train, dataset_labels_train)

    # Prediction
    test_prediction = knc.predict(dataset_proj_test)
    print("Test Prediction: ", test_prediction)

    # Accuracy
    accuracy = accuracy_score(dataset_labels_test, test_prediction)
    print("Accuracy: ", accuracy)

    # Save Model in Binary
    knc_pickle = open("pca_model", "wb")
    pickle.dump(knc, knc_pickle)
    knc_pickle.close()

    # Save Top Vectors
    np.savez("top_eigenvectors.npz", top_eigenvectors)

    # Images and Stuff
    if True:
        # Mean Image of every single digit
        num_col = 5
        num_row = 2

        fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
        for index, image in enumerate(range(10)):
            ax = axes[image // num_col, image % num_col]
            ax.imshow(data_mean_images[image], cmap="gray")
            ax.set_title("Label: {}".format(index))
        plt.tight_layout()
        plt.show()

        # Mean Image - all digits
        plt.imshow(data_mean_image)
        plt.show()

        # Plot Weight -> ~20 - 150
        plt.plot([i for i in range(eigenvalues_cov.shape[0])], list_eigenvectors_weight)
        plt.show()
