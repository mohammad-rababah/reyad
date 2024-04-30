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

def multinomial_naive_bayes(X_train, y_train, X_test):
    class_counts = np.bincount(y_train.flatten())
    total_classes = len(class_counts)
    class_probs = class_counts / len(y_train)
    feature_counts = np.zeros((total_classes, X_train.shape[1]))

    for c in range(total_classes):
        class_indices = (y_train == c).flatten()
        feature_counts[c] = np.sum(X_train[class_indices], axis=0)

    feature_probs = (feature_counts + 1) / (np.sum(feature_counts, axis=1, keepdims=True) + X_train.shape[1])

    class_probs = np.log(class_probs)
    feature_probs = np.log(feature_probs)

    pred_probs = np.dot(X_test, feature_probs.T) + class_probs

    return np.argmax(pred_probs, axis=1).reshape(-1, 1)


# Load training data
X_train, y_train = load_data("training_spam.csv")

# Load test data
X_test, y_test = load_data("testing_spam.csv")

# Predict using Multinomial Naive Bayes
y_pred = multinomial_naive_bayes(X_train, y_train, X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

