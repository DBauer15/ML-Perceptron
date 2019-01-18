import numpy as np


def perc(w, X):
    """
    Predict the class for feature vector X using weights w
    :param w: weight vector (homogenized)
    :param X: feature vector (homogenized)
    :return: Prediction for class1 (1) or class2 (-1)
    """
    total = 0

    for i in range(len(X)):
        total += w[i] * X[i]

    return np.sign(total)


def percTrain(X, t, maxIts, online):
    """
    Delegates to the appropriate training algorithm depending on "online"
    :param X: Input features
    :param t: Input labels
    :param maxIts: Maximum number of iterations if algorithm does not terminate first
    :param online: Determines the algorithm to be used
    :return: The trained weight vector w
    """
    print("0% Done")
    if online:
        return train_online(X, t, maxIts)
    else:
        return train_batch(X, t, maxIts)


def train_online(X, t, maxIts):
    """
    Online training algorithm as seen in lecture materials "MLVC_part1.pdf" page 70
    :param X: Input features
    :param t: Input labels
    :param maxIts: Maximum number of iteration
    :return: Trained weight vector w
    """
    all_correct = False

    # Initialize all weights with zero
    # Size same as single feature vector since it was already homogenized
    w = np.zeros(len(X[1, :]))

    # Iteration counter
    its = 0

    # This loops through "epochs"
    # One epoch looks at all feature vectors in X
    while not all_correct and its < maxIts:
        # Assume all correct until error occurs
        all_correct = True

        # Iterate over all feature vectors in X
        for i in range(len(X)):
            # Normalize the feature by it's outcome (see lecture materials)
            xt = np.multiply(X[i], t[i])

            # Check if the dot between current w and normalized feature is positive
            # If it is, that's fine - we can go on
            # If not, we need to adjust the weights to "include" this feature in the positive R3 space
            if np.dot(w.transpose(), xt) <= 0:
                # One error means the current epoch is not "all correct"
                all_correct = False
                # Adjust position of w by the normalized feature
                w = np.add(w, xt)
        its += 1
        if (its/float(maxIts))*100 % 10 == 0:
            print("{:.2f}% Done".format((its/float(maxIts))*100))

    print("All done")
    return w


def train_batch(X, t, maxIts):
    """
    Online training algorithm as seen in lecture materials "MLVC_part1.pdf" page 70
    :param X: Input features
    :param t: Input labels
    :param maxIts: Maximum number of iteration
    :return: Trained weight vector w
    """
    all_correct = False

    # Initialize all weights with zero
    # Size same as single feature vector since it was already homogenized
    w = np.zeros(len(X[1, :]))

    # Iteration counter
    its = 0

    while not all_correct and its < maxIts:
        # Declare a delta-W (dw) in which to store the gradient
        dw = np.zeros(len(X[1, :]))

        # Assume all correct until error occurs
        all_correct = True

        # Iterate over all feature vectors in X
        for i in range(len(X)):
            # Normalize the feature by it's outcome (see lecture materials)
            xt = np.multiply(X[i], t[i])

            # Check if the dot between current w and normalized feature is positive
            # If it is, that's fine - we can go on
            # If not, we need to adjust the weights to "include" this feature in the positive R3 space
            if np.dot(w.transpose(), xt) <= 0:
                # One error means the current epoch is not "all correct"
                all_correct = False
                # Adjust position of w by the normalized feature
                dw = np.add(dw, xt)

        # Adjust w by the gradient delta-W (dw)
        w = np.add(w, dw)

        its += 1
        if (its/float(maxIts))*100 % 10 == 0:
            print("{:.2f}% Done".format((its/float(maxIts))*100))

    print("All done")
    return w
