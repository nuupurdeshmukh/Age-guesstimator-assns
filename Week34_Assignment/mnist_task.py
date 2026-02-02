import numpy as np
from tensorflow.keras.datasets import mnist


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], -1).T / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).T / 255.0

    return x_train, y_train, x_test, y_test


def init_network():
    W1 = np.random.randn(10, 784) * 0.01
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) * 0.01
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2


def relu(Z):
    return np.maximum(0, Z)


def softmax(Z):
    exp_z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def forward_propagation(W1, b1, W2, b2, X):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def one_hot_encode(Y):
    one_hot = np.zeros((Y.size, 10))
    one_hot[np.arange(Y.size), Y] = 1
    return one_hot.T


def backward_propagation(Z1, A1, A2, W2, X, Y):
    m = Y.size
    one_hot_y = one_hot_encode(Y)

    dZ2 = A2 - one_hot_y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * (Z1 > 0)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2


def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return W1, b1, W2, b2


def train(X, Y, epochs, lr):
    W1, b1, W2, b2 = init_network()

    for i in range(epochs):
        Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_propagation(Z1, A1, A2, W2, X, Y)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)

        if i % 100 == 0:
            preds = np.argmax(A2, axis=0)
            acc = np.mean(preds == Y)
            print(f"Epoch {i}: Accuracy = {acc * 100:.2f}%")

    return W1, b1, W2, b2


if __name__ == "__main__":

    X_train, Y_train, X_test, Y_test = load_data()

    W1, b1, W2, b2 = train(X_train, Y_train, 1000, 0.1)

    np.savez("mnist_weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)

    print("\nTraining complete.")

    _, _, _, A2_test = forward_propagation(W1, b1, W2, b2, X_test)
    acc = np.mean(np.argmax(A2_test, axis=0) == Y_test)

    print(f"Test Accuracy: {acc * 100:.2f}%")
