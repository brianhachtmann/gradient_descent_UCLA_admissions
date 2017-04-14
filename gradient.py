import numpy as np
from data_prep import training_features, training_targets, test_features, test_targets

def sigmoid(h):
    # return the sigmoid of h
    return 1/(1+np.exp(-h))


# seed np.random with same from data_prep.py for debug simplicity
np.random.seed(42)

# neural network size:
# calculate number of training records and features
n_records, n_input = training_features.shape
n_layer1 = 2
n_output = training_targets.ndim

# neural network hyper parameters
epochs = 10000 # 1000 is a good start here 7000 seemed to converge for the single layer rev
learning_rate = 0.5 / n_records # TODO: why 0.5 / n_records here? to average?

# init weights from normal distribution scaled to 1/root(len(n_input)):
in_weights_l1 = np.random.normal(scale=1/n_input**0.5, size = (n_input, n_layer1))

# init weights from normal distribution scaled to 1/root(len(n_input))
l1_weights_out = np.random.normal(scale=1/n_layer1**0.5, size = (n_layer1, n_output))
# print("weights = {0}".format(weights))

# for mean squared error reduction tracking
# last_mse = None

# train:
for e in range(epochs):
    # calculate gradient descent step
    in_grad_l1 = np.zeros(in_weights_l1.shape)
    l1_grad_out = np.zeros(l1_weights_out.shape)
    for x, y in zip(training_features.values, training_targets):
        # calculate network output for this record
        lin_out = sigmoid(np.dot(x, in_weights_l1))
        l1_out  = sigmoid(np.dot(lin_out, l1_weights_out))
        # print("lin_out.shape = ",lin_out.shape)
        # print("l1_out.shape = ",l1_out.shape)

        # backpropogation
        # calculate gradient for layer
        l1_grad_out += (y - l1_out) * (l1_out * (1 - l1_out)) * l1_out
        in_grad_l1 += np.dot(l1_weights_out, y - l1_out) * (lin_out * (1-lin_out)) * x[:,None]
    # sum weight steps to calculate new weight
    in_weights_l1 += learning_rate * in_grad_l1
    l1_weights_out += learning_rate * l1_grad_out

    # # track descension and print mean squared error
    # training_output = sigmoid(np.dot(sigmoid(np.dot(training_features, in_weights_l1)),l1_weights_out))
    # # print(training_output[:,0])
    # # print(training_targets.values)
    # mse = np.mean((training_output[:,0] - training_targets.values) ** 2)
    # if last_mse and mse < last_mse:
    #      print("training mse:", mse)
    # else:
    #      print("training mse:", mse, "WARNING: mse increasing")

    # # upate mse for next lap.
    # last_mse = mse

# test:
# calculate accuracy on test data:
test_predictions = sigmoid(np.dot(sigmoid(np.dot(test_features, in_weights_l1)),l1_weights_out)) > 0.5
accuracy = np.mean(test_predictions[:,0] == test_targets.values)
print("Admissions prediction test accuracy = ", accuracy)



















