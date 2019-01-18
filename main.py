import util
from mnist_loader import MNISTLoader
import perceptron as prct


def main(modality, epochs, online):
    # Declare the number classes to be used in this task (0-9)
    num1 = 3
    num2 = 0

    # Instantiate mnist loader
    loader = MNISTLoader()

    # Task 1.1.1
    # Load training and test data
    img_train, lbl_train = loader.load_training_data(num1, num2, 1000)
    img_test, lbl_test = loader.load_testing_data(num1, num2, 500)

    # Calculate features
    feat_train = util.get_image_props(img_train)
    feat_trans_train = util.get_feature_transform(feat_train)
    feat_full_train = util.get_augmented_image_vectors(img_train)
    feat_test = util.get_image_props(img_test)
    feat_trans_test = util.get_feature_transform(feat_test)
    feat_full_test = util.get_augmented_image_vectors(img_test)

    if modality & REGIONPROPS != 0:
        util.plot(feat_train, lbl_train, 'r', 'c', None, "Filled Area & Convex Area Features")

    # Task 1.1.2

    if modality & W_2 != 0:
        # Train and test on 2 features
        w_2 = prct.percTrain(feat_train, lbl_train, epochs, online)
        acc_w_2 = util.evaluate(feat_test, lbl_test, w_2, "2 features")
        util.plot(feat_train, lbl_train, 'r', 'c', w_2, "Features: 2, Epochs: {}, Online: {}, Accuracy: {:.2f}%".format(epochs, online, acc_w_2))

    if modality & W_5 != 0:
        # Train and test on 5 features
        w_5 = prct.percTrain(feat_trans_train, lbl_train, epochs, online)
        acc_w_5 = util.evaluate(feat_trans_test, lbl_test, w_5, "5 features")
        util.plot(feat_trans_train, lbl_train, 'r', 'c', w_5, "Features: 5, Epochs: {}, Online: {}, Accuracy: {:.2f}%".format(epochs, online, acc_w_5))

    if modality & W_784 != 0:
        # Train and test on 28x28 features
        w_784 = prct.percTrain(feat_full_train, lbl_train, epochs, online)
        acc_w_784 = util.evaluate(feat_full_test, lbl_test, w_784, "28x28 features")
        util.visualize_w(w_784, "Features: 28x28, Epochs: {}, Online: {}, Accuracy: {:.2f}%".format(epochs, online, acc_w_784))


REGIONPROPS = 0x0001
W_2         = 0x0010
W_5         = 0x0100
W_784       = 0x1000

main(W_784, 20, True)
