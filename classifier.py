import numpy as np
import os
import cv2
from sklearn import svm
from sklearn.metrics import classification_report

class Classifier:
    def __init__(self):
        '''
        if use_chdir:
            os.chdir(curr_dir)
        '''
        train_labels = self.readLabels("training_data.txt")
        # print(len(train_labels))
        train_images = self.readImages()
        # print(len(train_images))
        train_images = self.convertImagesTo2D(train_images)

        '''
        validation_labels = self.readLabels(("validation_data.txt"))
        print(len(validation_labels))
        validation_images = self.readImages2()
        print(len(validation_images))
        validation_images = self.convertImagesTo2D(validation_images)
        '''

        self.classifier = svm.SVC(kernel = 'rbf', C=4)
        self.classifier.fit(train_images, train_labels)
        
        '''
        predictions = self.classifier.predict(validation_images)
        print(classification_report(validation_labels, predictions))
        '''

    # Function designed to read the images from a file given by <path> and return them as a numpy array
    def readImages(self):
        images = []
        for i in range(580):
            aux_file_name = "./training_data/filled_title_" + str(i) + ".png"
            aux_image = cv2.imread(aux_file_name, 0)
            aux_image = cv2.resize(aux_image, (75, 75))
            images.append(aux_image)

        images = np.array(images)
        return images

    '''
    def readImages2(self):
        images = []
        for i in range(479, 580):
            aux_file_name = "./validation_data/filled_title_" + str(i) + ".png"
            aux_image = cv2.imread(aux_file_name, 0)
            aux_image = cv2.resize(aux_image, (75, 75))
            images.append(aux_image)

        images = np.array(images)
        return images
    '''

    # Function designed to read the labels from a file where, on each line, the first 10 characters represent
    # the number of the image and the 11-th character is it's classification (0..9).
    def readLabels(self, path):
        labels = list()
        file = open(path)
        for eachLine in file:
            for char in eachLine:
                labels.append(int(char))
        labels = np.array(labels)
        labels = labels.astype(np.int8)
        file.close()
        return labels


    def convertImagesTo2D(self, images):
        # Because the array is 3D, I need to reshape it into a 2D one in order for the classification to work.
        nr_of_images, x_axis, y_axis = images.shape
        images = images.reshape((nr_of_images, x_axis * y_axis))
        return images

    def predict(self, image):
        image = cv2.resize(image, (75, 75))
        x_axis, y_axis = image.shape
        image = image.reshape((1, x_axis * y_axis))
        return self.classifier.predict(image)
        


if __name__ == "__main__":
    clasificator = Classifier()