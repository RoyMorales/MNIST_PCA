# Main PCA Digitos


# Imports
import matplotlib.pyplot as plt
import numpy as np
import gzip

def read_dataset_images_train(num_images):
    image_dataset_train = gzip.open("./Dataset/train-images-idx3-ubyte.gz")
    image_size = 28

    image_dataset_train.read(16)
    buffer = image_dataset_train.read(image_size * image_size * num_images)
    data_image_train = np.frombuffer(buffer, dtype = np.uint8)#.astype(np.float32)
    data_image_train = data_image_train.reshape(num_images, image_size, image_size)
    
    return data_image_train

def read_dataset_labels_train(num_labels):
    labels_dataset_train = gzip.open("./Dataset/train-labels-idx1-ubyte.gz")
    labels_dataset_train.read(8)

    buffer = labels_dataset_train.read(num_labels)
    labels = np.frombuffer(buffer, dtype = np.uint8)#.astype(np.uint64)
    
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


def principal_component(image):
    print("ToDo!")




if __name__ == "__main__":
    dataset_size = 60000

    data_image_train = read_dataset_images_train(dataset_size)
    data_labels_train = read_dataset_labels_train(dataset_size)

    print("Train Dataset Images Shape: ", data_image_train.shape)
    print("Train Dataset Labels Shape: ", data_labels_train.shape)
    
    data_mean_images = average_number(data_image_train, data_labels_train)
    data_mean_image = np.mean(data_mean_images, axis=0)

    # Mean Image of every single digit
    num_col = 5
    num_row = 2

    fig, axes = plt.subplots(num_row, num_col, figsize= (1.5 * num_col, 2 * num_row))
    for index, image in enumerate(range(10)):
        ax = axes[image // num_col, image % num_col]
        ax.imshow(data_mean_images[image], cmap= "gray")
        ax.set_title('Label: {}'.format(index))
    plt.tight_layout()
    plt.show()

    # Mean Image - all digits
    plt.imshow(data_mean_image)
    plt.show()


