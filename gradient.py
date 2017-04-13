import numpy as np
from data_prep import training_features, training_targets, test_features, test_targets

def sigmoid(h):
    # return the sigmoid of h
    return 1/(1+np.exp(-h))

# replaced by output * (1 - output) in del_weights calc in train
# def sigmoiddot(h):
#     # return the derivative of sigmoid h
#      s = sig(h)
#      return s*(1-s)

# seed np.random with same from data_prep.py for debug simplicity
# np.random.seed(42)

# calculate number of training records and features
n_records, n_features = training_features.shape

# neural network hyper parameters
epochs = 7000 # 1000 is a good start here
learning_rate = 0.5 / n_records # TODO: why 0.5 / n_records here? to average?

# init weights from normal distribution scaled to 1/root(len(features))
weights = np.random.normal(scale=1/n_features**0.5, size = n_features)
# print("weights = {0}".format(weights))

# for mean squared error reduction tracking
last_mse = None

# train:
for e in range(epochs):
    # calculate gradient descent step
    grad_desc = np.zeros(weights.shape)
    for x, y in zip(training_features.values, training_targets):
        # calculate network output for this record
        output = sigmoid(np.dot(x, weights))
        # print("network output {0}".format(y_hat))

        # calculate the error for this record
        error = y - output
        # print("error = {0}".format(error))

        # sum ( error * sigmoid gradient at h * x )
        grad_desc += error * (output * (1 - output)) * x
    # sum weight steps to calculate new weight
    weights += learning_rate * grad_desc

    # track descension and print mean squared error
    training_output = sigmoid(np.dot(training_features, weights))
    mse = np.mean((training_output - training_targets) ** 2)
    if last_mse and mse < last_mse:
         print("training mse:", mse)
    else:
         print("training mse:", mse, "WARNING: mse increasing")

    # upate mse for next lap.
    last_mse = mse

# test:
# calculate accuracy on test data:
test_predictions = sigmoid(np.dot(test_features, weights)) > 0.5
accuracy = np.mean(test_predictions == test_targets)
print("Admissions prediction test accuracy = ", accuracy)



















