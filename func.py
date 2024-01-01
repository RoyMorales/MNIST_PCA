# Functions

# Imports 
import gzip
import numpy as np

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


def scatter_matrix(matrix):
    scatter_matrix = np.zeros((matrix.shape[1], matrix.shape[1]))

    for i in range(matrix.shape[0]):
        scatter_matrix += np.outer(matrix[i], matrix[i])

    return scatter_matrix

def euclidean_distance(point_a, point_b):
    return np.linalg.norm(point_a - point_b)

def mahalanobis_distance(point_a, point_b, inv_cov_matrix):
    diff = point_a - point_b
    return np.sqrt(np.dot(np.dot(diff.T, inv_cov_matrix), diff))
