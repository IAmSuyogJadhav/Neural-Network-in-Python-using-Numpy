import time
import numpy as np
np.random.seed(777)
start = time.time()


# ------------------ Data
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]])  # Test data
y = np.array([[1], [0], [0], [1], [1], [0], [1]])  # Test labels
m, n = x.shape


# ------------------ Neural Network Parameters
a2_nodes = 20
a3_nodes = 10
a4_nodes = 1
max_iter = 10000
alpha = 0.3
lamda = 0
j_history = np.zeros(max_iter)

w1 = np.random.randn(a2_nodes, n+1)
w2 = np.random.randn(a3_nodes, a2_nodes+1)
w3 = np.random.randn(a4_nodes, a3_nodes+1)
# WHEN ADDING MORE LAYERS IN FUTURE, FOLLOW THE ABOVE PATTERN.
# NOTE THAT THE BIAS TERMS WILL BE ADDED IN THE TRAIN FUNCTION.
# SO, NO NEED TO ADD THEM IN THE X BEFORE TRAINING.


def sigmoid(z):
    """Calculates the sigmoid activation function."""

    return 1 / (1 + np.exp(-z))


def cost(x, y, w1, w2, w3, lamda=0):
    """Calculates the cost for the neural network."""
    m, n = x.shape
    a1 = np.append(np.ones(m).reshape(m, 1), x, axis=1)
    z2 = a1.dot(w1.T)
    a2 = np.append(np.ones(z2.shape[0]).reshape(z2.shape[0], 1), sigmoid(z2), axis=1)
    z3 = a2.dot(w2.T)
    a3 = np.append(np.ones(z3.shape[0]).reshape(z3.shape[0], 1), sigmoid(z3), axis=1)
    z4 = a3.dot(w3.T)
    a4 = sigmoid(z4)

    j = (-1/m)*np.sum(y.T.dot(np.log(a4)) + (1 - y).T.dot(np.log(1 - a4)))\
        + (lamda/(2*m))*(np.sum(w1[:, 1:]**2) + np.sum(w2[:, 1:]**2) + np.sum(w3[:, 1:]**2))
    return j


def train(x, y, w1, w2, w3, alpha=0.01, lamda=0):
    """Trains the neural network. Performs one pass each of forward propagation and Back propagation."""

    m, n = x.shape
    a1 = np.append(np.ones(m).reshape(m, 1), x, axis=1)
    z2 = a1.dot(w1.T)
    a2 = np.append(np.ones(z2.shape[0]).reshape(z2.shape[0], 1), sigmoid(z2), axis=1)
    z3 = a2.dot(w2.T)
    a3 = np.append(np.ones(z3.shape[0]).reshape(z3.shape[0], 1), sigmoid(z3), axis=1)
    z4 = a3.dot(w3.T)
    a4 = sigmoid(z4)

# ------------------ Backprop
    del4 = a4 - y
    del3 = (del4.dot(w3)) * a3 * (1 - a3)
    # THE LAST AND SECOND LAST DEL TERMS WILL ALWAYS BE LIKE AS SHOWN ABOVE
    del2 = (del3[:, 1:].dot(w2)) * a2 * (1 - a2)
    # WHEN ADDING MORE LAYERS IN FUTURE, REPEAT THE SAME THING AS ABOVE
    # FOR ALL THE DEL TERMS EXCEPT THE SECOND LAST AND LAST ONE

    w1_grad = (1/m) * (del2[:, 1:].T.dot(a1))
    w2_grad = (1/m) * (del3[:, 1:].T.dot(a2))
    # WHEN ADDING MORE LAYERS IN FUTURE, REPEAT THE SAME THING AS ABOVE
    # FOR ALL THE GRAD TERMS EXCEPT THE LAST ONE
    w3_grad = (1/m) * (del4.T.dot(a3))
    # THE LAST GRAD TERM WILL ALWAYS BE LIKE AS SHOWN ABOVE

    w1_reg = (lamda/m) * w1
    w2_reg = (lamda/m) * w2
    w3_reg = (lamda/m) * w3
    w1_reg[:, 0] = 0
    w2_reg[:, 0] = 0
    w3_reg[:, 0] = 0

# ------------------ Update Parameters
    w1 = w1 - alpha * w1_grad - w1_reg
    w2 = w2 - alpha * w2_grad - w2_reg
    w3 = w3 - alpha * w3_grad - w3_reg

    return w1, w2, w3


def predict(x, w1, w2, w3, threshold=0.5):
    """Predicts the outcome of a trained neural network for given data."""

    m, n = x.shape
    a1 = np.append(np.ones(m).reshape(m, 1), x, axis=1)
    z2 = a1.dot(w1.T)
    a2 = np.append(np.ones(z2.shape[0]).reshape(z2.shape[0], 1), sigmoid(z2), axis=1)
    z3 = a2.dot(w2.T)
    a3 = np.append(np.ones(z3.shape[0]).reshape(z3.shape[0], 1), sigmoid(z3), axis=1)
    z4 = a3.dot(w3.T)
    a4 = sigmoid(z4)
    a4[a4 < threshold] = 0
    a4[a4 >= threshold] = 1
    return a4


print("\n**********************************************\n")
print("Training...")
# ------------------ Training
for i in range(max_iter):
    w1, w2, w3 = train(x, y, w1, w2, w3, alpha=0.01, lamda=0)
    j = cost(x, y, w1, w2, w3, lamda=lamda)
    print("\r" + "{}% |".format(int(100 * i / max_iter) + 1) + '#' * int((int(100 * i / max_iter) + 1) / 5) +
          ' ' * (20 - int((int(100 * i / max_iter) + 1) / 5)) + '|',
          end="") if not i % (max_iter / 100) else print("", end="")
    j_history[i] = j


print("\n**********************************************\n")
# ------------------ Analysis
train_prediction = predict(x, w1, w2, w3)
print("Accuracy on the training set= ", 100*np.sum(train_prediction == y)/m)

end = time.time()
print("Execution Time= %0.2f s" % (end - start))
print("\n**********************************************\n")
