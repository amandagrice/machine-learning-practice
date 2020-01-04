# Machine Learning Practice

I'm teaching myself the basics behind machine learning. This repo will be the dumping ground for my notes, toy projects, and things I do while following tutorials. It won't contain anything really original, but I'll have it on my github in case it helps any other learners. 

<img src="./giphy.gif" width="250">

## Table of Contents

1. [MNIST Digits](https://github.com/amandagrice/machine-learning-practice/tree/master/MNIST%20Digits) - A neural net that can read handwritten numbers. 
2. [IMDB Classification](https://github.com/amandagrice/machine-learning-practice/tree/master/IMDB%20Classification) - A neural net that classifies movie reviews from [IMDB](https://www.imdb.com/) as positive or negative.
3. [Reuters Classification](https://github.com/amandagrice/machine-learning-practice/tree/master/Reuters%20Classification) - A neural net that can label Reuters newswires with one of 46 topics. 

---

## Notes (in progress)

- **artificial intelligence** - computers solving problems generally associated with the human mind

- **machine learning** - the study of algorithms and statistical models that computers use to perform a task without specific instructions but relying on statistics and inference instead. Subset of the field of artificial intelligence. 
  - classical programming: data + rules = answers
  - machine learning: data + answers = rules

- **deep learning** - subset of machine learning. The “deep” part refers to the idea of successive layers of representations. Through machine learning, computers “learn” to represent data throw multiple stages - progressively extracting higher level features from raw input. For example, a deep learning algorithm that takes in images might first recognize edges or colors in lower layers, and then later identify objects in the images. 

- **shallow learning** - other methods of machine learning that don’t rely on layers
  - classical machine learning approaches (not deep learning):
    - probabilistic modeling - application of principles of statistics to data analysis
      - Naive Bayes
      - logistic regression (logreg)
    - kernel methods
      - support vector machine (SVM)
        - decision boundaries
        - separation hyperplane
        - maximizing the margin
      - kernel trick
      - kernel function
    - decision trees
      - Random Forest
      - gradient boosting machines

- **symbolic AI** - main focus of AI from 1950s to 1980s. Researchers tried to represent problems using a series of human-readable rules. 
  - **expert systems** - form of symbolic AI; a computer system that emulates the decision-making ability of a human expert. Generally, tons of if-then code.

- **analytical engine** - a general-purpose mechanical computer designed by Charles Babbage. Never finished. 

- **Turing test** - a test of a machine’s ability to exhibit intelligent behavior indistinguishable from a human. Proposed by Alan Turing in 1950. Basically a human evaluator has a conversation with two entities over a text chat - one robot and one human and if the evaluator can’t reliably determine who is the robot - the robot passes.

- hypothesis space
- neural networks
  - convolutional neural networks (convnets)
- weights
  - the transformation implemented by a layer of a NN is parameterized by its weights
  - learning is finding the correct values for weights
- loss function (aka objective function)
- optimizer
- backpropagation
- training loop
- feature engineering
- Kaggle - https://www.kaggle.com/
- Tensor
- Tensor Processing Unit (TPU)
- CPU
- GPU
- Moore’s Law
