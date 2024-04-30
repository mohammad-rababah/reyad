import numpy as np


def load_data(file_path):
    data = np.loadtxt(open(file_path), delimiter=",").astype(int)
    y = data[:, 0].reshape(-1, 1)
    X = data[:, 1:]
    return X, y


def accuracy_score(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = 2 * np.random.random((input_size, hidden_size)) - 1
        self.weights_hidden_output = 2 * np.random.random((hidden_size, output_size)) - 1

    def train(self, X_train, y_train, learning_rate=0.1, epochs=1000):
        for epoch in range(epochs):
            # Forward propagation
            hidden_layer_input = np.dot(X_train, self.weights_input_hidden)
            hidden_layer_output = sigmoid(hidden_layer_input)

            output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
            output = sigmoid(output_layer_input)

            # Backpropagation
            error = y_train - output
            delta_output = error * sigmoid_derivative(output)

            error_hidden = delta_output.dot(self.weights_hidden_output.T)
            delta_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)

            # Update weights
            self.weights_hidden_output += hidden_layer_output.T.dot(delta_output) * learning_rate
            self.weights_input_hidden += X_train.T.dot(delta_hidden) * learning_rate

    def predict(self, X_test):
        hidden_layer_input = np.dot(X_test, self.weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        output = sigmoid(output_layer_input)

        return output

# Load training data
X_train, y_train = load_data("training_spam.csv")

# Load test data
X_test, y_test = load_data("testing_spam.csv")

# Create and train the model
input_size = X_train.shape[1]
hidden_size = 4
output_size = 1
model = NeuralNetwork(input_size, hidden_size, output_size)
model.train(X_train, y_train)

y_pred = np.round(model.predict(X_test))

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)