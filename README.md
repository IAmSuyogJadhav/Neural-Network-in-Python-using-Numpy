# Neural-Network-in-Python-using-Numpy
This is the implementation of a fully customizable neural network with arbitrary no. of hidden layers using naught but NumPy, in Python.

# Documentation for the train module
- ```NN```: A Network that uses Sigmoid activation function. Methods defined here:
  - ```add_layer(self, n_nodes, output_layer=False)``` : Adds a layer of specified no. of output nodes. For the output layer, the flag  ```output_layer``` must be True. A network must have an output layer.
  - ```sigmoid(self, z)```: Calculates the sigmoid activation function.
  - ```predict(self, x, predict=True, argmax=True, rand_weights=False)```: Performs a pass of forwrd propagation on `x`. If ```predict``` is set to ```True```, trained weights are used and predictions are returned in a single vector(if `argmax` is set to `True`) with labels from 0 to (`n_classes` - 1).
  - `cost(self, x, y, lamda=0)`: Calculates the cost for given data and labels, with trained weights.
  - `fit(self, data, labels, test=[], test_labels=[], alpha=0.01, lamda=0, epochs=50)`: Performs specified no. of epoches. One epoch = one pass of forward propagation + one pass of backpropagation.
- `plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r)`: Used to plot confusion matrix. (Internal function)
