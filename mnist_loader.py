from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split


class MNISTLoader:
    __mnist = None
    __train_img = None
    __test_img = None
    __train_lbl = None
    __test_lbl = None

    def __init__(self):
        self.__load_mnist()

    def __load_mnist(self):
        self.__mnist = fetch_mldata('MNIST original')
        self.__train_img, self.__test_img, self.__train_lbl, self.__test_lbl = train_test_split(
            self.__mnist.data, self.__mnist.target, test_size=1 / 2.0, random_state=0)

    def load_training_data(self, num1, num2, size):
        images = []
        labels = []

        i = 0
        while len(images) < size:
            if self.__train_lbl[i] == num1 or self.__train_lbl[i] == num2:
                images.append(self.__train_img[i])
                labels.append(1 if self.__train_lbl[i] == num1 else -1)
            i = i + 1

        return images, labels

    def load_testing_data(self, num1, num2, size):
        images = []
        labels = []

        i = 0
        while len(images) < size:
            if self.__test_lbl[i] == num1 or self.__test_lbl[i] == num2:
                images.append(self.__test_img[i])
                labels.append(1 if self.__test_lbl[i] == num1 else -1)
            i = i + 1

        return images, labels
