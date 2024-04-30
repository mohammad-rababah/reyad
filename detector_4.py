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


class LinearSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        y_ = np.where(y <= 0, -1, 1)  # Convert labels to -1 and 1

        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.learning_rate * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.weights) - self.bias
        return np.sign(approx)


# Load training data
X_train, y_train = load_data("training_spam.csv")

# Load test data
X_test, y_test = load_data("testing_spam.csv")

# Ensure y_train and y_test are column vectors
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Create and train the model
svm = LinearSVM()
svm.fit(X_train, y_train)

# Predict using the model
y_pred = svm.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
