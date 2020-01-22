# MNIST Digits

This is a neural network to identify handwritten digits created using the [Keras](https://keras.io/) deep learning library on top of TensorFlow. It was trained using the [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/). 

![MNIST Data Sample](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

I used Francois Chollet's book "Deep Learning with Python" as a tutorial. 

There are 2 versions:
- mnist_digits.py has 2 regular dense layers and has accuracy of 98.2%
- mnist_digits_convnet.py uses a convolutional neural network and has accuracy of 99.2%