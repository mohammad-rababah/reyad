import numpy as np

# pre-processing -> train , test

# you try to approach 3 algorithms
# talk about presicion and accuracy
# cinfusion matrix  -> 2x2

# 1. Single-layer neurone
# 2. Naive Bayes
# 3. Neural Network

# Load data
def load_data(file_path):
    data = np.loadtxt(open(file_path), delimiter=",").astype(int)
    y = data[:, 0].reshape(-1, 1)
    X = data[:, 1:]
    return X, y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Accuracy calculation
def accuracy_score(y_true, y_pred): # 1000 --> 800 800/1000 = 0.8
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy


# Precision calculation
def precision_score(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    precision = true_positives / (true_positives + false_positives)
    return precision


# Single-layer Neural Network
class SingleLayerNN:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size, 1) # this create array with dim (input_size ,1)
        self.bias = np.random.randn(1) # 1*1

    def train(self, X_train, y_train, learning_rate=0.01, epochs=1000):
        for epoch in range(epochs):
            output = self.predict(X_train)
            error = y_train - output
            # update weights w1 = w1 + learning_rate * x1 * error
            self.weights += learning_rate * X_train.T.dot(error)
            self.bias += learning_rate * np.sum(error)

    def predict(self, X_test):
        # x1*w1 + x2*w2 + x3*w3 + ... + xn*wn + b = sum
        # sum > 0 .5-> 1
        return np.where(np.dot(X_test, self.weights) + self.bias > 0.5, 1, 0)


# Naive Bayes
class NaiveBayes:
    def fit(self, X_train, y_train):
        self.class_probs = {}
        self.feature_probs = {}

        classes = np.unique(y_train)
        for c in classes:
            class_indices = (y_train == c).flatten()
            self.class_probs[c] = np.mean(class_indices)
            self.feature_probs[c] = np.mean(X_train[class_indices], axis=0)

    def predict(self, X_test):
        pred_probs = {}
        for c, class_prob in self.class_probs.items():
            feature_prob = self.feature_probs[c]
            pred_probs[c] = np.prod(X_test * feature_prob + (1 - X_test) * (1 - feature_prob), axis=1) * class_prob
        return np.argmax(list(pred_probs.values()), axis=0)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # if you have 20 input  and 4

        self.weights_input_hidden = 2 * np.random.random((input_size, hidden_size)) - 1 # 20*4
        self.weights_hidden_output = 2 * np.random.random((hidden_size, output_size)) - 1 # 4*1

    def train(self, X_train, y_train, learning_rate=0.1, epochs=1000):
        for epoch in range(epochs):
            # Forward propagation
            hidden_layer_input = np.dot(X_train, self.weights_input_hidden)
            hidden_layer_output = sigmoid(hidden_layer_input)

            output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
            output = sigmoid(output_layer_input)

            # Backpropagation
            error = y_train - output # 1000 - 1000
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

# Ensure y_train and y_test are column vectors
y_train = y_train.reshape(-1, 1) # from vector(n,) to matrix (20,n) * (n,1)
y_test = y_test.reshape(-1, 1)

# Single-layer Neural Network
nn = SingleLayerNN(X_train.shape[1])
nn.train(X_train, y_train)
nn_pred = nn.predict(X_test)

nn_accuracy = accuracy_score(y_test, nn_pred)
nn_precision = precision_score(y_test, nn_pred)

# Naive Bayes
nb = NaiveBayes()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_pred = nb_pred.reshape(-1, 1)
nb_accuracy = accuracy_score(y_test, nb_pred)
nb_precision = precision_score(y_test, nb_pred)

# NNM
input_size = X_train.shape[1]
hidden_size = 4
output_size = 1
model = NeuralNetwork(input_size, hidden_size, output_size)
model.train(X_train, y_train)


nnmuralNet_pred = np.round(model.predict(X_test))
nnmuralNet_accuracy = accuracy_score(y_test, nnmuralNet_pred)
nnmuralNet_precision = precision_score(y_test, nnmuralNet_pred)
# Print results
print("Single-layer Neural Network:") # 1
print("Accuracy:", nn_accuracy)
print("Precision:", nn_precision)

print("\nNaive Bayes:") # 1
print("Accuracy:", nb_accuracy)
print("Precision:", nb_precision)

print("\nNeural Network:") #  0
print("Accuracy:", nnmuralNet_accuracy)
print("Precision:", nnmuralNet_precision)

sum_pred = nn_pred.reshape(-1, 1) + nb_pred.reshape(-1, 1) + nnmuralNet_pred.reshape(-1, 1)

# Convert sum_pred to binary where 1 indicates a sum of 2 or more
final_pred = (sum_pred >= 2).astype(int)

# Calculate accuracy and precision for the final prediction
final_accuracy = accuracy_score(y_test, final_pred)
final_precision = precision_score(y_test, final_pred)

print("\nFinal Prediction:")
print("Accuracy:", final_accuracy)
print("Precision:", final_precision)
