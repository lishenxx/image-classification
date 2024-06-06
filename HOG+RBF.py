import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from skimage.feature import hog
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

def extract_features(images):
    descriptors_list = []
    for image in images:
        features = hog(image, orientations=8, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
        descriptors_list.append(features)
    return np.array(descriptors_list)

train_features = extract_features(train_images)
test_features = extract_features(test_images)

clf = make_pipeline(StandardScaler(), svm.SVC(kernel='rbf', C=1))
clf.fit(train_features, train_labels)

predicted_labels = clf.predict(test_features)
print("Classification report for classifier %s:\n%s\n"
      % (clf, classification_report(test_labels, predicted_labels)))
print("Classification accuracy:", accuracy_score(test_labels, predicted_labels))
