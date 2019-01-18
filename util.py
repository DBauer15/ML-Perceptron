import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label
from skimage.measure import regionprops
import perceptron as prct


def get_image_props(images):
    # Calculate image properties for all training images
    props = np.zeros((len(images), 3))
    for i in range(len(images)):
        img = np.array(images[i], dtype="float")
        # Binarize image for better processing
        img = img > 120
        # Reshape as 2d matrix
        img = img.reshape((28, 28))
        # Label the image and extract features
        label_img = label(img, connectivity=img.ndim)
        prop = regionprops(label_img)
        # Use two prominent features to optimally distinguish numbers
        # Add 1 in the first position for augmentation
        # TODO: Maybe find better features?
        props[i, :] = (1, prop[0].filled_area, prop[0].convex_area)
    return props


def get_feature_transform(features):
    # Calculate (1, x1, x2) -> (1, x1, x2, x1^2, x2^2, x1x2)
    props = np.zeros((len(features), 6))
    for i in range(len(features)):
        x1 = features[i, 1]
        x2 = features[i, 2]
        props[i, :] = (1, x1, x2, x1*x1, x2*x2, x1*x2)

    return props


def get_augmented_image_vectors(images):
    props = np.zeros((len(images), 785))
    for i in range(len(images)):
        img = images[i].tolist()
        img.insert(0, 1)
        props[i, :] = img

    return props


def show_images(images):
    for i in images:
        img = np.array(i, dtype="float")
        img = img.reshape((28, 28))
        plt.imshow(img, cmap="gray")
        plt.show()


def plot(props, training_labels, col1, col2, w=None, title=None):
    # Create a color vector according to the labels
    colors = [(col1 if (item == 1) else col2) for item in training_labels]

    if w is not None:
        num_features = len(w)-1
        x_min = int(np.amin(props[:, 1])-1)
        x_max = int(np.amax(props[:, 1])+1)
        y_min = int(np.amin(props[:, 2])-1)
        y_max = int(np.amax(props[:, 2])+1)
        x = np.linspace(x_min, x_max, x_max-x_min)
        y = np.linspace(y_min, y_max, y_max-y_min)

        fig, ax = plt.subplots()
        Z = np.zeros((x_max-x_min, y_max-y_min))
        for i in range(len(Z[:, 0])):
            for j in range(len(Z[0, :])):
                k = i+x_min
                l = j+y_min
                if num_features == 2:
                    Z[i, j] = prct.perc(w, [1, k, l])
                if num_features == 5:
                    Z[i, j] = prct.perc(w, [1, k, l, k**2, l**2, k*l])
        Z = Z.transpose()
        ax.contour(x, y, Z, colors='k', linewidths=0.2)

    # Plot also the training points
    plt.scatter(props[:, 1], props[:, 2], color=colors, s=0.75)
    plt.title(title)
    plt.show()


def visualize_w(w, title="W Visualization"):
    w = w[1:]
    w = np.reshape(w, (28, 28))
    plt.imshow(w, cmap="gray")
    plt.title(title)
    plt.show()


def evaluate(props, labels, w, title="weights"):
    num_all = len(props)
    num_correct = 0

    for i in range(num_all):
        if prct.perc(w, props[i]) == labels[i]:
            num_correct = num_correct + 1

    percent_correct = (num_correct/float(num_all))*100
    print("Evaluation result for {}: {} of {} instances predicted correctly ({:.2f}%)"
           .format(title, num_correct, num_all, percent_correct))

    return percent_correct
