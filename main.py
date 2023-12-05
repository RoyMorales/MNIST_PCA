# Main PCA Digitos


# Imports
import matplotlib.pyplot as plt
import numpy as np
import gzip

# Machine Learning Imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def read_dataset_images_train(num_images):
    image_dataset_train = gzip.open("./Dataset/train-images-idx3-ubyte.gz")
    image_size = 28

    image_dataset_train.read(16)
    buffer = image_dataset_train.read(image_size * image_size * num_images)
    data_image_train = np.frombuffer(buffer, dtype=np.uint8)  # .astype(np.float32)
    data_image_train = data_image_train.reshape(num_images, image_size, image_size)

    return data_image_train


def read_dataset_images_test(num_images):
    image_dataset_test = gzip.open("./Dataset/t10k-images-idx3-ubyte.gz")
    image_size = 28

    image_dataset_test.read(16)
    buffer = image_dataset_test.read(image_size * image_size * num_images)
    data_image_test = np.frombuffer(buffer, dtype=np.uint8)  # .astype(np.float32)
    data_image_test = data_image_test.reshape(num_images, image_size, image_size)

    return data_image_test


def read_dataset_labels_train(num_labels):
    labels_dataset_train = gzip.open("./Dataset/train-labels-idx1-ubyte.gz")
    labels_dataset_train.read(8)

    buffer = labels_dataset_train.read(num_labels)
    labels = np.frombuffer(buffer, dtype=np.uint8)  # .astype(np.uint64)

    return labels


def read_dataset_labels_test(num_labels):
    labels_dataset_test = gzip.open("./Dataset/t10k-labels-idx1-ubyte.gz")
    labels_dataset_test.read(8)

    buffer = labels_dataset_test.read(num_labels)
    labels = np.frombuffer(buffer, dtype=np.uint8)  # .astype(np.uint64)

    return labels


def average_number(data_image, data_labels):
    # Elements in data
    size_data_elements = [0 for i in range(10)]

    # Alocation array of 9 matrix
    sum_array = np.zeros((10, 28, 28))
    mean_array = np.zeros((10, 28, 28))

    for index, image in enumerate(data_image):
        label_number = np.asarray(data_labels[index])
        sum_array[label_number] += image
        size_data_elements[label_number] += 1

    print("Number Counter: ", size_data_elements)

    for number in range(len(sum_array)):
        mean_array[number] = sum_array[number] / size_data_elements[number]

    return mean_array


def mean_vector_row(matrix):
    mean_vector = np.mean(matrix, axis=1)
    return mean_vector


def scatter_matrix(matrix, mean_vector):
    scatter_matrix = np.zeros((28, 28))

    for i in range(matrix.shape[1]):
        scatter_matrix += (matrix[:, i].reshape(28, 1) - mean_vector).dot(
            (matrix[:, i].reshape(28, 1) - mean_vector).T
        )

    return scatter_matrix


def principal_component(image):
    print("ToDo!")


if __name__ == "__main__":
    dataset_size = 60000
    dataset_size_test = 10000

    # Read Dataset
    data_image_train = read_dataset_images_train(dataset_size)
    data_labels_train = read_dataset_labels_train(dataset_size)
    data_image_test = read_dataset_images_test(dataset_size_test)
    data_labels_test = read_dataset_labels_test(dataset_size_test)

    print("Train Dataset Images Shape: ", data_image_train.shape)
    print("Train Dataset Labels Shape: ", data_labels_train.shape)

    # Mean of the images from the dataset - Train
    data_mean_images = average_number(data_image_train, data_labels_train)
    data_mean_image = np.mean(data_mean_images, axis=0)

    # Mean of the images from the dataset - Test
    data_mean_images_test = average_number(data_image_test, data_labels_test)
    data_mean_image_test = np.mean(data_mean_images_test, axis=0)

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

    # Mean Vetor of Matrix
    mean_vector = mean_vector_row(data_mean_image)
    print("Mean Vector: ", mean_vector.shape)

    # Scatter Matrix
    scatter_matrix = scatter_matrix(data_mean_image, mean_vector)
    print("Scatter Matrix: ", scatter_matrix.shape)

    # Conv Matrix
    cov_matrix = np.cov(data_mean_image)
    print("Cov Matrix: ", cov_matrix.shape)

    # Eigenvalues and Eigenvectors
    eigenvalues_sc, eigenvectors_sc = np.linalg.eig(scatter_matrix)
    eigenvalues_cov, eigenvectors_cov = np.linalg.eig(cov_matrix)
    print("Eigenvalues Scatter: ", eigenvalues_sc.shape)
    print("Eigenvectors Scatter: ", eigenvectors_sc.shape)
    print("Eigenvalues Cov: ", eigenvalues_cov.shape)
    print("Eigenvectors Cov: ", eigenvectors_cov.shape)

    # Sort Eigenvalues - Eigenvectors in descending order
    sorted_ind_sc = np.argsort(eigenvalues_sc)
    eigenvalues_sc = eigenvalues_sc[sorted_ind_sc]
    eigenvectors_sc = eigenvectors_sc[sorted_ind_sc]

    sorted_ind_cov = np.argsort(eigenvalues_cov)
    eigenvalues_cov = eigenvalues_cov[sorted_ind_cov]
    eigenvectors_cov = eigenvectors_cov[sorted_ind_cov]

    print("Sorted Index SC: ", sorted_ind_sc)
    print("Sorted Index Cov: ", sorted_ind_cov)

    # MISSING Eigenvalues Weight

    number_eignvectors = 2
    top_eignvectors = eigenvectors_cov[:, 0:number_eignvectors]
    print("Top Eignvectors: ", top_eignvectors.shape)

    # Project the dataset onto the Eigenvectors
    dataset_proj_train = np.dot(data_mean_image, top_eignvectors)
    dataset_proj_test = np.dot(data_mean_image_test, top_eignvectors)
    print("Dataset Proj Train: ", dataset_proj_train.shape)
    print("Dataset Proj Test: ", dataset_proj_test.shape)

    print("Data Test Label: ", data_labels_test.shape)
    print("Data Train Label: ", data_labels_train.shape)
    # ToDo! -> Classifier
    knc = KNeighborsClassifier(n_neighbors=3)
    knc.fit(dataset_proj_train, data_labels_train)

    # Prediction
    test_prediction = knc.predict(dataset_proj_test)
    print("Test Prediction: ", test_prediction)

    # Accuracy
    accuracy = accuracy_score(data_labels_test, test_prediction)
    print("Accuracy: ", accuracy)



