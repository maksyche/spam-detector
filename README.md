# Spam Detector

- This is a very basic SMS spam recognition ANN written in Python for learning purposes. It uses 3 dense layers (input,
  hidden, output), [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) activation function, **Glorot** weight
  initialization, [Stochastic Gradient Descend (SGD)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
  optimizer, and [Mean Squared Error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error) loss function. Check out
  [this page](https://github.com/maksyche/useful-theory/blob/master/machine-learning/README.md#neural-networks) for
  complete mathematics of this network.
- Word embeddings are not trainable; used pre-trained [GloVe embeddings](https://github.com/stanfordnlp/GloVe)
  (100-dimensional version).
- 2 model implementations: [Tensorflow](./spam_detector_tensorflow.py) and [custom](./spam_detector.py). The goal of the 
  custom implementation was to make it as readable as possible. There's no optimization, and the network is VERY slow.
- A [dataset](./dataset.csv) with 5k messages (2.5k ham and 2.5k spam) included. The dataset is created with generative
  AI and modified manually (you may still find some weird messages).
- ~97% success rate with the included dataset ([pre-trained custom model attached](./model%5B2300,%20128%5D.json)).

