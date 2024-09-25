import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Loading and preparing train and test datasets
train_data = np.loadtxt('mnist_train.csv', delimiter=',', skiprows=1, dtype=np.float32)
test_data = np.loadtxt('mnist_test.csv', delimiter=',', skiprows=1, dtype=np.float32)

x_train = train_data[:, 1:] / 255.0
y_train = train_data[:, 0].astype(int)

x_test = test_data[:, 1:] / 255.0
y_test = test_data[:, 0].astype(int)


def ReLU(x):
    return np.maximum(0, x)


def deriv_ReLU(x):
    return (x > 0).astype(float)


def softmax(x):
    ex = np.exp(x - np.max(x, axis=1, keepdims=True))
    return ex / np.sum(ex, axis=1, keepdims=True)


def one_hot(label, n_class=10):
    return np.eye(n_class)[label]


def train(x, y, alpha, epochs):
    # Initializing weights and biases
    input_size = 784
    hidden_size = 10
    output_size = 10

    hidden_weights = np.random.randn(input_size, hidden_size)
    output_weights = np.random.randn(hidden_size, output_size)

    hidden_bias = np.zeros((1, hidden_size))
    output_bias = np.zeros((1, output_size))

    for index in tqdm(range(epochs), desc='Trainning model'):
        # Forward pass
        z1 = ReLU(np.dot(x, hidden_weights) + hidden_bias)
        z2 = softmax(np.dot(z1, output_weights) + output_bias)

        y_one_hot = one_hot(y)

        error = z2 - y_one_hot

        # Backpropagation
        delta_output_weights = np.dot(z1.T, error) / x.shape[0]
        delta_output_bias = np.sum(error, axis=0, keepdims=True) / x.shape[0]
        
        error_hidden = np.dot(error, output_weights.T) * deriv_ReLU(z1)
        delta_hidden_weights = np.dot(x.T, error_hidden) / x.shape[0]
        delta_hidden_bias = np.sum(error_hidden, axis=0, keepdims=True) / x.shape[0]

        # Updating parameters
        output_weights -= alpha * delta_output_weights
        output_bias -= alpha * delta_output_bias
        hidden_weights -= alpha * delta_hidden_weights
        hidden_bias -= alpha * delta_hidden_bias

    return hidden_weights, output_weights, hidden_bias, output_bias


def predict(x, params):
    z1 = ReLU(np.dot(x, params[0]) + params[2])
    z2 = softmax(np.dot(z1, params[1]) + params[3])
    return np.argmax(z2)


def get_accuracy(x, y, params):
    z1 = ReLU(np.dot(x, params[0]) + params[2])
    z2 = softmax(np.dot(z1, params[1]) + params[3])

    preds = np.argmax(z2, axis=1)

    true_preds = np.sum(preds == y)
    return true_preds / x.shape[0]


# Training the model
model = train(x_train, y_train, 0.1, 500)

# Predicting a sample from the test set
index = 32
sample_test = x_test[index].reshape(1, -1)
prediction = predict(sample_test, model)

print(f'Predicted label: {prediction}\nTrue label: {y_test[index]}')

# Displaying the sample image
plt.gray()
plt.imshow(sample_test.reshape((28, 28)) * 255.0)
plt.show()
