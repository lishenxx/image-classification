import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from cyvlfeat.sift import dsift
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

def extract_features(images):
    descriptors_list = []
    for image in images:
        image = np.squeeze(image) 
        _, descriptors = dsift(image, step=[5, 5], fast=True)
        if descriptors is not None:
            descriptors_list.append(descriptors.mean(axis=0))
        else:
            descriptors_list.append(np.zeros(128))
    return np.array(descriptors_list)

train_features = extract_features(train_images.reshape(-1, 28, 28)) 
test_features = extract_features(test_images.reshape(-1, 28, 28))

clf = make_pipeline(StandardScaler(), svm.SVC(kernel='linear',C=1))
clf.fit(train_features, train_labels)

from sklearn.metrics import classification_report, accuracy_score

predicted_labels = clf.predict(test_features)
print("Classification report for classifier %s:\n%s\n"
      % (clf, classification_report(test_labels, predicted_labels)))
print("Classification accuracy:", accuracy_score(test_labels, predicted_labels))
