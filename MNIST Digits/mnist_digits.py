from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

training_images = training_images.reshape((60000, 28 * 28))
training_images = training_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

training_labels = to_categorical(training_labels)
test_labels = to_categorical(test_labels)

network.fit(training_images, training_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)

print('\ntest_acc:', test_acc)  # 0.9824



