import numpy as np
import matplotlib.pyplot as plt
import mplcyberpunk

plt.rc('figure', figsize=(12, 8))
plt.rcParams['figure.dpi'] = 100

plt.style.use('cyberpunk')

train_data = np.loadtxt('mnist_train.csv', delimiter=',', skiprows=1, dtype=np.float32)
test_data = np.loadtxt('mnist_test.csv', delimiter=',', skiprows=1, dtype=np.float32)

x_train = train_data[:, 1:] / 255.0
y_train = train_data[:, 0].astype(int)

x_test = test_data[:, 1:] / 255.0
y_test = test_data[:, 0].astype(int)


def leaky_ReLU(x, alpha=0.05):
    return np.where(x > 0, x, alpha * x)


def deriv_leaky_ReLU(x, alpha=0.05):
    return np.where(x > 0, 1, alpha)


def softmax(x):
    ex = np.exp(x - np.max(x, axis=1, keepdims=True))
    return ex / np.sum(ex, axis=1, keepdims=True)


def one_hot(label, n_class=10):
    return np.eye(n_class)[label]


def get_accuracy(x, y, params):
    z1 = leaky_ReLU(np.dot(x, params[0]) + params[4])
    z2 = leaky_ReLU(np.dot(z1, params[1]) + params[5])
    z3 = softmax(np.dot(z2, params[2]) + params[3])

    preds = np.argmax(z3, axis=1)
    true_preds = np.sum(preds == y)
    return true_preds / x.shape[0]


def cross_entropy_loss(y_true, y_pred):
    n_samples = y_pred.shape[0]
    log_p = - np.log(y_pred[range(n_samples), y_true] + 1e-15)
    loss = np.sum(log_p) / n_samples
    return loss


def train(x, y, alpha, epochs, hidden_size1, hidden_size2, X_test, Y_test):
    input_size = 784
    output_size = 10

    hidden_weights1 = np.random.randn(input_size, hidden_size1)
    hidden_bias1 = np.zeros((1, hidden_size1))

    hidden_weights2 = np.random.randn(hidden_size1, hidden_size2)
    hidden_bias2 = np.zeros((1, hidden_size2))

    output_weights = np.random.randn(hidden_size2, output_size)
    output_bias = np.zeros((1, output_size))

    accuracies, epoch_list, losses = [], [], []

    for index in range(epochs):
        z1 = leaky_ReLU(np.dot(x, hidden_weights1) + hidden_bias1)
        z2 = leaky_ReLU(np.dot(z1, hidden_weights2) + hidden_bias2)
        z3 = softmax(np.dot(z2, output_weights) + output_bias)

        y_one_hot = one_hot(y)
        error = z3 - y_one_hot

        delta_output_weights = np.dot(z2.T, error) / x.shape[0]
        delta_output_bias = np.sum(error, axis=0, keepdims=True) / x.shape[0]

        error_hidden2 = np.dot(error, output_weights.T) * deriv_leaky_ReLU(z2)
        delta_hidden_weights2 = np.dot(z1.T, error_hidden2) / x.shape[0]
        delta_hidden_bias2 = np.sum(error_hidden2, axis=0, keepdims=True) / x.shape[0]

        error_hidden1 = np.dot(error_hidden2, hidden_weights2.T) * deriv_leaky_ReLU(z1)
        delta_hidden_weights1 = np.dot(x.T, error_hidden1) / x.shape[0]
        delta_hidden_bias1 = np.sum(error_hidden1, axis=0, keepdims=True) / x.shape[0]

        output_weights -= alpha * delta_output_weights
        output_bias -= alpha * delta_output_bias
        hidden_weights2 -= alpha * delta_hidden_weights2
        hidden_bias2 -= alpha * delta_hidden_bias2
        hidden_weights1 -= alpha * delta_hidden_weights1
        hidden_bias1 -= alpha * delta_hidden_bias1

        if (index + 1) % 10 == 0:
            accuracy = get_accuracy(x_test, y_test, [hidden_weights1, hidden_weights2, output_weights, output_bias, hidden_bias1, hidden_bias2])
            accuracies.append(accuracy)
            epoch_list.append(index + 1)

            z1_test = leaky_ReLU(np.dot(x_test, hidden_weights1) + hidden_bias1)
            z2_test = leaky_ReLU(np.dot(z1_test, hidden_weights2) + hidden_bias2)
            z3_test = softmax(np.dot(z2_test, output_weights) + output_bias)

            loss = cross_entropy_loss(Y_test, z3_test)
            losses.append(loss)

            print(f'Epoch {index + 1}\nAccuracy -> {accuracy * 100:.2f}%\nLoss -> {loss:.4f}\n--------------------\n')

    return hidden_weights1, hidden_weights2, output_weights, hidden_bias1, hidden_bias2, output_bias, accuracies, epoch_list, losses


def predict(x, params):
    z1 = leaky_ReLU(np.dot(x, params[0]) + params[4])
    z2 = leaky_ReLU(np.dot(z1, params[1]) + params[5])
    z3 = softmax(np.dot(z2, params[2]) + params[3])
    return np.argmax(z3)


learning_rate = 0.1
hidden_1_size = 128
hidden_2_size = 64

train_hidden_weights1, train_hidden_weights2, train_output_weights, train_hidden_bias1, train_hidden_bias2, train_output_bias, train_accuracy, train_epochs, train_losses = train(
    x_train, y_train, alpha=learning_rate, epochs=100, hidden_size1=hidden_1_size, hidden_size2=hidden_2_size, X_test=x_test, Y_test=y_test
)

plt.title(f'Accuracy over epochs with {hidden_1_size} and {hidden_2_size} hidden neurons')
plt.xlabel('Epochs')
plt.ylabel('Accuracy % (x100)')
plt.plot(train_epochs, train_accuracy, color='green')
mplcyberpunk.add_glow_effects()
plt.show()
