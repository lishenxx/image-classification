import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

def extract_features(images):
    descriptors_list = []
    for image in images:
        features = hog(image, orientations=8, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
        descriptors_list.append(features)
    return np.array(descriptors_list)

train_features = extract_features(train_images)
test_features = extract_features(test_images)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

model = Sequential()
model.add(Dense(128, input_dim=train_features.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_features, train_labels, epochs=20, batch_size=64, validation_data=(test_features, test_labels))

loss, accuracy = model.evaluate(test_features, test_labels)
print(f"Test Accuracy: {accuracy}")

predicted_labels = np.argmax(model.predict(test_features), axis=1)
true_labels = np.argmax(test_labels, axis=1)

print("Classification report:\n", classification_report(true_labels, predicted_labels))
print("Classification accuracy:", accuracy_score(true_labels, predicted_labels))

