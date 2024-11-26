import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pylab as pylab
import numpy as np

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, X_test.shape)  # (60000, 28, 28) (10000, 28, 28)

# Reshape to [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# Normalize training and test data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]  # Number of categories


def FC_model():
    """Create and return a fully connected model."""
    # Create model
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))  # Flatten input image to 1D vector
    model.add(Dense(200, activation='relu'))  # First hidden layer with 200 units
    model.add(Dropout(0.15))  # Add dropout with 15% rate
    model.add(Dense(200, activation='relu'))  # Second hidden layer with 200 units
    model.add(Dense(num_classes, activation='softmax'))  # Output layer for classification
    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Build the model
model = FC_model()
model.summary()

# Fit the model
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=10,
          batch_size=200,
          verbose=2)

# Evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: {:.4f} \nError: {:.4f}".format(scores[1], 100 - scores[1] * 100))

# Visualize the weights of the first dense layer
W = model.get_layer('dense').get_weights()  # Get the weights of the first dense layer
print(W[0].shape)  # Shape of weight matrix
print(W[1].shape)  # Shape of bias vector

# Visualizing weights of the first 200 hidden units
fig = pylab.figure(figsize=(20, 20))
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95, hspace=0.05, wspace=0.05)

pylab.gray()  # Set color to gray

# Loop through the first 200 neurons and visualize their weights
for i in range(200):
    # Reshape the weights to be 28x28 (reshape each column of W[0] into a 28x28 image)
    pylab.subplot(15, 14, i + 1)
    pylab.imshow(np.reshape(W[0][:, i], (28, 28)))  # Reshape each column to 28x28 for visualization
    pylab.axis('off')  # Hide axis

# Title for the plot
pylab.suptitle('Dense Layer Weights (200 hidden units)', size=20)
pylab.show()