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
    # mean_vector = mean_vector_row(data_mean_image)
    # print("")
    # print("Mean Vector: ", mean_vector.shape)

    # Scatter Matrix
    # scatter_matrix = scatter_matrix(data_mean_image, mean_vector)
    # print("Scatter Matrix: ", scatter_matrix.shape)

    # Conv Matrix
    cov_matrix = np.cov(data_images_center_train, rowvar=False)
    print("Cov Matrix: ", cov_matrix.shape)

    # Eigenvalues and Eigenvectors
    # eigenvalues_sc, eigenvectors_sc = np.linalg.eig(scatter_matrix)
    # Tipa para complexo 0j. -> Não sei 
    eigenvalues_cov, eigenvectors_cov = np.linalg.eig(cov_matrix)

    print("")
    # print("Eigenvalues Scatter: ", eigenvalues_sc.shape)
    # print("Eigenvectors Scatter: ", eigenvectors_sc.shape)
    print("Eigenvalues Cov: ", eigenvalues_cov.shape)
    print("Eigenvectors Cov: ", eigenvectors_cov.shape)

    # Sort Eigenvalues - Eigenvectors in descending order
    # sorted_ind_sc = np.argsort(eigenvalues_sc)
    # eigenvalues_sc = eigenvalues_sc[sorted_ind_sc]
    # eigenvectors_sc = eigenvectors_sc[sorted_ind_sc]

    sorted_ind_cov = np.argsort(eigenvalues_cov)[::-1]
    eigenvalues_cov = eigenvalues_cov[sorted_ind_cov]
    eigenvectors_cov = eigenvectors_cov[:, sorted_ind_cov]

    # print("Sorted Index SC: ", sorted_ind_sc)
    # print("Sorted Index Cov: ", sorted_ind_cov)

    # MISSING Eigenvalues Weight

    number_eignvectors = 8
    top_eignvectors = eigenvectors_cov[:, :number_eignvectors]
    print("")
    print("Top Eignvectors: ", top_eignvectors.shape)

    # Project the dataset onto the Eigenvectors
    # Não sei porque esta a tipar para complexo 0j.
    dataset_proj_train = (np.dot(data_images_center_train, top_eignvectors)).astype(
        "float32"
    )
    dataset_proj_test = (np.dot(data_images_center_test, top_eignvectors)).astype(
        "float32"
    )
    print("")
    print("Dataset Proj Train: ", dataset_proj_train.shape)
    print("Dataset Proj Test: ", dataset_proj_test.shape)

    # ToDo! -> Classifier
    knc = KNeighborsClassifier(n_neighbors=10)
    knc.fit(dataset_proj_train, dataset_labels_train)

    # Prediction
    test_prediction = knc.predict(dataset_proj_test)
    print("Test Prediction: ", test_prediction)

    # Accuracy
    accuracy = accuracy_score(dataset_labels_test, test_prediction)
    print("Accuracy: ", accuracy)
