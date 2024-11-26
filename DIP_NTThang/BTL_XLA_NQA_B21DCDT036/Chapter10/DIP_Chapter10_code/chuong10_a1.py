import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pylab
from tensorflow.keras.datasets import mnist

# Load dữ liệu
(X_train, y_train), (X_test, y_test) = mnist.load_data()
np.random.seed(0)
train_indices = np.random.choice(60000, 50000, replace=False)
valid_indices = [i for i in range(60000) if i not in train_indices]
X_valid, y_valid = X_train[valid_indices, :, :], y_train[valid_indices]
X_train, y_train = X_train[train_indices, :, :], y_train[train_indices]

image_size = 28
num_labels = 10

# Hàm tiền xử lý dữ liệu
def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32) / 255.0
    labels = tf.keras.utils.to_categorical(labels, num_labels)
    return dataset, labels

X_train, y_train = reformat(X_train, y_train)
X_valid, y_valid = reformat(X_valid, y_valid)
X_test, y_test = reformat(X_test, y_test)

print('Training set', X_train.shape)
print('Validation set', X_valid.shape)
print('Test set', X_test.shape)

# Định nghĩa mô hình
batch_size = 256
num_hidden_units = 1024
lambda1 = 0.1
lambda2 = 0.1

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(image_size * image_size,)),
    tf.keras.layers.Dense(num_hidden_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lambda1)),
    tf.keras.layers.Dense(num_labels, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(lambda2))
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.008),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Huấn luyện mô hình
history = model.fit(X_train, y_train, epochs=10, batch_size=batch_size, validation_data=(X_valid, y_valid))

# Hiển thị trực quan trọng số
weights = model.layers[0].get_weights()[0]
plt.figure(figsize=(18, 18))
indices = np.random.choice(num_hidden_units, 225)
for j in range(225):
    plt.subplot(15, 15, j + 1)
    plt.imshow(weights[:, indices[j]].reshape((image_size, image_size)), cmap='gray')
    plt.xticks([], [])
    plt.yticks([], [])
plt.title('Weights Visualization')
plt.show()

# Vẽ đồ thị accuracy và loss
atr = history.history['accuracy']
av = history.history['val_accuracy']
ll = history.history['loss']

pylab.figure(figsize=(8, 12))
pylab.subplot(211)
pylab.plot(range(len(atr)), atr, '.-', label='Training Accuracy')
pylab.plot(range(len(av)), av, '.-', label='Validation Accuracy')
pylab.xlabel('Epochs')
pylab.ylabel('Accuracy')
pylab.legend(loc='lower right')

pylab.subplot(212)
pylab.plot(range(len(ll)), ll, '.-', label='Training Loss')
pylab.xlabel('Epochs')
pylab.ylabel('Loss')
pylab.legend(loc='upper right')

pylab.show()