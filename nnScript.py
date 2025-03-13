import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    sig = []

    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


"""
    if np.isscalar(z):
        return 1 / (1 + np.exp(-z))

    for i in z:
        # if i >= 0:
        val = 1 / (1 + np.exp(-i))
        # else:
        # val = np.exp(i) / (1 + np.exp(i))
        sig.append(val)
    return np.array(sig)  # your code here
"""


def preprocess():
    """Input:
    Although this function doesn't have any input, you are required to load
    the MNIST data set from file 'mnist_all.mat'.

    Output:
    train_data: matrix of training set. Each row of train_data contains
      feature vector of a image
    train_label: vector of label corresponding to each image in the training
      set
    validation_data: matrix of training set. Each row of validation_data
      contains feature vector of a image
    validation_label: vector of label corresponding to each image in the
      training set
    test_data: matrix of training set. Each row of test_data contains
      feature vector of a image
    test_label: vector of label corresponding to each image in the testing
      set

    Some suggestions for preprocessing step:
    - feature selection"""

    mat = loadmat("mnist_all.mat")  # loads the MAT object as a Dictionary
    total_data = []
    for i in range(10):
        total_data.append(np.vstack([mat[f"train{i}"]]))
    total_label = []
    for i in range(10):
        total_label.append(np.hstack([np.full(mat[f"train{i}"].shape[0], i)]))

    total_data = np.vstack(total_data)
    total_label = np.hstack(total_label)

    samples = total_data.shape[0]
    indices = np.random.permutation(samples)
    validation_indices = indices[50000:]
    training_indices = indices[:50000]

    train_data = total_data[training_indices]
    train_label = total_label[training_indices]

    validation_data = total_data[validation_indices]
    validation_label = total_label[validation_indices]

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples.
    # Your code here.

    # Feature selection
    # Your code here.
    feature_selected = np.std(train_data, axis=0) > 0
    train_data = train_data[:, feature_selected]
    validation_data = validation_data[:, feature_selected]

    print("preprocess done")

    return (
        train_data,
        train_label,
        validation_data,
        validation_label,
        # test_data,
        # test_label,
    )


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.


    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0 : n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)) :].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    n = training_data.shape[0]
    training_data = np.hstack((training_data, np.ones((n, 1))))

    z = np.dot(training_data, w1.T)
    z = sigmoid(z)
    z = np.hstack((z, np.ones((n, 1))))

    o = np.dot(z, w2.T)
    o = sigmoid(o)

    # what i did here was get the input and output layer activations^
    y = np.zeros((n, n_class))
    for i in range(n):
        y[i, int(training_label[i])] = 1

    # this is for one hot encoding  ^

    original_error = -np.sum(y * np.log(o) + (1 - y) * np.log(1 - o)) / n
    w1_sum_squares = np.sum(w1[:, :n_input] ** 2)
    w2_sum_squares = np.sum(w2[:, :n_hidden] ** 2)

    regularization = (lambdaval / (2 * n)) * (w1_sum_squares + w2_sum_squares)

    obj_val = original_error + regularization

    delta_o = o - y

    delta_z = (1 - z) * z * np.dot(delta_o, w2)

    delta_z = delta_z[:, :-1]

    gradient_w1 = np.dot(delta_z.T, training_data) / n
    gradient_w2 = np.dot(delta_o.T, z) / n

    gradient_w1[:, :-1] += (lambdaval / n) * w1[:, :-1]
    gradient_w2[:, :-1] += (lambdaval / n) * w2[:, :-1]

    # computing gradients and adding regulaiztion term^
    # we do not want to include the bias term ^
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((gradient_w1.flatten(), gradient_w2.flatten()), 0)

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""

    # labels = np.array([])
    # Your code here

    data = np.hstack((data, np.ones((data.shape[0], 1))))
    net_first = np.dot(data, np.transpose(w1))
    s = sigmoid(net_first)

    s = np.hstack((s, np.ones((s.shape[0], 1))))
    net_second = np.dot(s, np.transpose(w2))

    final_s = sigmoid(net_second)
    labels = np.argmax(final_s, axis=1)
    return labels


"""**************Neural Network Script Starts here********************************"""
if __name__ == "__main__":

    (
        train_data,
        train_label,
        validation_data,
        validation_label,
        # test_data,
        # test_label,
    ) = preprocess()

    #  Train Neural Network

    # set the number of nodes in input unit (not including bias unit)
    n_input = train_data.shape[1]

    # set the number of nodes in hidden unit (not including bias unit)
    n_hidden = 83

    # set the number of nodes in output unit
    n_class = 10

    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

    # set the regularization hyper-parameter
    lambdaval = 55

    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

    opts = {"maxiter": 50}  # Preferred value.

    nn_params = minimize(
        nnObjFunction, initialWeights, jac=True, args=args, method="CG", options=opts
    )

    # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
    # and nnObjGradient. Check documentation for this function before you proceed.
    # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)

    # Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = nn_params.x[0 : n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)) :].reshape((n_class, (n_hidden + 1)))

    # Test the computed parameters

    predicted_label = nnPredict(w1, w2, train_data)

    # find the accuracy on Training Dataset

    print(
        "\n Training set Accuracy:"
        + str(100 * np.mean((predicted_label == train_label).astype(float)))
        + "%"
    )

    predicted_label = nnPredict(w1, w2, validation_data)

    # find the accuracy on Validation Dataset

    print(
        "\n Validation set Accuracy:"
        + str(100 * np.mean((predicted_label == validation_label).astype(float)))
        + "%"
    )

    # predicted_label = nnPredict(w1, w2, test_data)

    # find the accuracy on Validation Dataset
"""
    print(
        "\n Test set Accuracy:"
        + str(100 * np.mean((predicted_label == test_label).astype(float)))
        + "%"
    )
"""
