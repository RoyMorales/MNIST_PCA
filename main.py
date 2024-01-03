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
    
    # Covariance Matrix
    cov_matrix = np.cov(data_images_center_train, rowvar=False)
    print("Cov Matrix: ", cov_matrix.shape)
    
    # Eigenvalues and Eigenvectors
    # Não sei porque está a tipar para complexo 0j.
    eigenvalues_cov, eigenvectors_cov = np.linalg.eig(cov_matrix)
    eigenvalues_cov = eigenvalues_cov.astype("float32")
    eigenvectors_cov = eigenvectors_cov.astype("float32")

    print("")
    print("Eigenvalues Cov: ", eigenvalues_cov.shape)
    print("Eigenvectors Cov: ", eigenvectors_cov.shape)

    # Sort Eigenvalues - Eigenvectors in descending order
    sorted_ind_cov = np.argsort(eigenvalues_cov)[::-1]
    eigenvalues_cov = eigenvalues_cov[sorted_ind_cov]
    eigenvectors_cov = eigenvectors_cov[:, sorted_ind_cov]

    # Covariance and Scatter Matrix Trace
    cov_trace = np.trace(cov_matrix)
    sum_eigenvalues_cov = sum(eigenvalues_cov)
    print("")
    print("Cov Trace: ", cov_trace)
    print("Sum Eigenvalues Cov: ", sum_eigenvalues_cov)

    # Eigenvectors Weight
    number_eigenvectors = 16
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
    print("")

    # Euclidian Distance
    # KNeighborsClassifier
    knc_eucl = KNeighborsClassifier(n_neighbors=50, metric=euclidean_distance)
    knc_eucl.fit(dataset_proj_train, dataset_labels_train)

    # Prediction
    test_prediction_eucl = knc_eucl.predict(dataset_proj_test)
    print("Test Prediction Euclidian: ", test_prediction_eucl)

    # Accuracy
    accuracy_eucl = accuracy_score(dataset_labels_test, test_prediction_eucl)
    print("Accuracy Euclidian: ", accuracy_eucl)

    # Save Model in Binary
    knc_eucl_pickle = open("pca_model_eucl", "wb")
    pickle.dump(knc_eucl, knc_eucl_pickle)
    knc_eucl_pickle.close()

    
    # Mahalanobis Distance
    # PCA Cov Matrix
    cov_matrix_pca = np.cov(dataset_proj_train, rowvar=False)
    inv_cov_matrix_pca = np.linalg.inv(cov_matrix_pca)
    print("PCA Cov Matrix: ", cov_matrix_pca.shape)
    print("")

    # KNeighborsClassifier
    knc_maha = KNeighborsClassifier(n_neighbors=50, metric=mahalanobis_distance, metric_params={"inv_cov_matrix": inv_cov_matrix_pca})
    knc_maha.fit(dataset_proj_train, dataset_labels_train)

    # Prediction
    test_prediction_maha = knc_maha.predict(dataset_proj_test)
    print("Test Prediction Mahalanobis: ", test_prediction_maha)

    # Accuracy
    accuracy_maha = accuracy_score(dataset_labels_test, test_prediction_maha)
    print("Accuracy Mahalanobis: ", accuracy_maha)

    # Save Model in Binary
    knc_maha_pickle = open("pca_model_maha", "wb")
    pickle.dump(knc_maha, knc_maha_pickle)
    knc_maha_pickle.close()

    # Save Top Vectors
    np.savez("top_eigenvectors.npz", top_eigenvectors)


