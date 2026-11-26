import numpy as np
import os
import pickle

from exercise_code.networks.layer import affine_forward, affine_backward, Sigmoid, Tanh, LeakyRelu, Relu
from exercise_code.networks.base_networks import Network


class ClassificationNet(Network):
    """
    A fully-connected classification neural network with configurable 
    activation function, number of layers, number of classes, hidden size and
    regularization strength. 
    """

    def __init__(self, activation=Sigmoid, num_layer=2,
                 input_size=3 * 32 * 32, hidden_size=100,
                 std=1e-3, num_classes=10, reg=0, **kwargs):
        """
        :param activation: choice of activation function. It should implement
            a forward() and a backward() method.
        :param num_layer: integer, number of layers. 
        :param input_size: integer, the dimension D of the input data.
        :param hidden_size: integer, the number of neurons H in the hidden layer.
        :param std: float, standard deviation used for weight initialization.
        :param num_classes: integer, number of classes.
        :param reg: float, regularization strength.
        """
        super().__init__("cifar10_classification_net")

        self.activation = activation()
        self.reg_strength = reg

        self.cache = None

        self.memory = 0
        self.memory_forward = 0
        self.memory_backward = 0
        self.num_operation = 0

        # Initialize random gaussian weights for all layers and zero bias
        self.num_layer = num_layer
        self.std = std
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.reset_weights()

    def forward(self, X):
        """
        Performs the forward pass of the model.

        :param X: Input data of shape N x D. Each X[i] is a training sample.
        :return: Predicted value for the data in X, shape N x 1
                 1-dimensional array of length N with the classification scores.
        """

        self.cache = {}
        self.reg = {}
        X = X.reshape(X.shape[0], -1)
        # Unpack variables from the params dictionary
        for i in range(self.num_layer - 1):
            W, b = self.params['W' + str(i + 1)], self.params['b' + str(i + 1)]

            # Forward i_th layer
            X, cache_affine = affine_forward(X, W, b)
            self.cache["affine" + str(i + 1)] = cache_affine

            # Activation function
            X, cache_sigmoid = self.activation.forward(X)
            self.cache["sigmoid" + str(i + 1)] = cache_sigmoid

            # Store the reg for the current W
            self.reg['W' + str(i + 1)] = np.sum(W ** 2) * self.reg_strength

        # last layer contains no activation functions
        W, b = self.params['W' + str(self.num_layer)],\
               self.params['b' + str(self.num_layer)]
        y, cache_affine = affine_forward(X, W, b)
        self.cache["affine" + str(self.num_layer)] = cache_affine
        self.reg['W' + str(self.num_layer)] = np.sum(W ** 2) * self.reg_strength

        return y

    def backward(self, dy):
        """
        Performs the backward pass of the model.

        :param dy: N x 1 array. The gradient wrt the output of the network.
        :return: Gradients of the model output wrt the model weights
        """

        # Note that last layer has no activation
        cache_affine = self.cache['affine' + str(self.num_layer)]
        dh, dW, db = affine_backward(dy, cache_affine)
        self.grads['W' + str(self.num_layer)] = \
            dW + 2 * self.reg_strength * self.params['W' + str(self.num_layer)]
        self.grads['b' + str(self.num_layer)] = db

        # The rest sandwich layers
        for i in range(self.num_layer - 2, -1, -1):
            # Unpack cache
            cache_sigmoid = self.cache['sigmoid' + str(i + 1)]
            cache_affine = self.cache['affine' + str(i + 1)]

            # Activation backward
            dh = self.activation.backward(dh, cache_sigmoid)

            # Affine backward
            dh, dW, db = affine_backward(dh, cache_affine)

            # Refresh the gradients
            self.grads['W' + str(i + 1)] = dW + 2 * self.reg_strength * \
                                           self.params['W' + str(i + 1)]
            self.grads['b' + str(i + 1)] = db

        return self.grads

    def save_model(self):
        self.eval()
        directory = 'models'
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(model, open(directory + '/' + self.model_name + '.p', 'wb'))

    def get_dataset_prediction(self, loader):
        self.eval()
        scores = []
        labels = []
        
        for batch in loader:
            X = batch['image']
            y = batch['label']
            score = self.forward(X)
            scores.append(score)
            labels.append(y)
            
        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()

        return labels, preds, acc
    
    def eval(self):
        """sets the network in evaluation mode, i.e. only computes forward pass"""
        self.return_grad = False
        
        # Delete unnecessary caches, to mitigate a memory prolbem.
        self.reg = {}
        self.cache = {}
        
    def reset_weights(self):
        self.params = {'W1':self.std * np.random.randn(self.input_size, self.hidden_size),
                       'b1': np.zeros(self.hidden_size)}

        for i in range(self.num_layer - 2):
            self.params['W' + str(i + 2)] = self.std * np.random.randn(self.hidden_size,
                                                                  self.hidden_size)
            self.params['b' + str(i + 2)] = np.zeros(self.hidden_size)

        self.params['W' + str(self.num_layer)] = self.std * np.random.randn(self.hidden_size,
                                                                  self.num_classes)
        self.params['b' + str(self.num_layer)] = np.zeros(self.num_classes)

        self.grads = {}
        self.reg = {}
        for i in range(self.num_layer):
            self.grads['W' + str(i + 1)] = 0.0
            self.grads['b' + str(i + 1)] = 0.0
        


class MyOwnNetwork(ClassificationNet):
    """
    Your first fully owned network!
    
    You can define any arbitrary network architecture here!
    
    As a starting point, you can use the code from ClassificationNet above as 
    reference or even copy it to MyOwnNetwork, but of course you're also free 
    to come up with a complete different architecture and add any additional 
    functionality! (Without renaming class functions though ;))
    """

    def __init__(self, activation=Relu, num_layer=2,
                 input_size=3 * 32 * 32, hidden_size=100,
                 std=None, num_classes=10, reg=0, **kwargs):
        """
        Your network initialization. For reference and starting points, check
        out the classification network above.

        Same architecture as ClassificationNet (affine -> activation -> ...
        -> affine), with two differences:
          - default activation is Relu instead of Sigmoid
          - std=None (default) uses He initialization: every layer gets its
            own std = sqrt(2 / fan_in), so std no longer has to be tuned.
            Pass a number for std to get the old fixed-std behaviour.
        """

        ########################################################################
        # TODO:  Your initialization here                                      #
        ########################################################################

        # The parent stores all settings (self.activation, self.reg_strength,
        # self.num_layer, self.hidden_size, self.std, ...) and then calls
        # self.reset_weights(). Because we override reset_weights() below,
        # Python calls OUR version, so the weights get the He initialization.
        super().__init__(activation=activation, num_layer=num_layer,
                         input_size=input_size, hidden_size=hidden_size,
                         std=std, num_classes=num_classes, reg=reg)
        self.model_name = "cifar10_my_own_network"

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def reset_weights(self):
        """
        Creates W1, b1, ..., WL, bL with He initialization.

        Why He initialization? If the weights are too small (e.g. std=1e-3),
        the signal shrinks a little in every layer and the gradients become
        tiny. If they are too large, it blows up. For ReLU, the variance stays
        roughly constant from layer to layer when Var(W) = 2 / fan_in, where
        fan_in is the number of inputs of that layer (Kaiming He et al. 2015).
        For the first layer: sqrt(2 / 3072) ~ 0.026.
        """
        # Width of every layer, e.g. num_layer=3: [3072, H, H, 10]
        layer_sizes = ([self.input_size]
                       + [self.hidden_size] * (self.num_layer - 1)
                       + [self.num_classes])

        self.params = {}
        for i in range(1, self.num_layer + 1):
            fan_in, fan_out = layer_sizes[i - 1], layer_sizes[i]
            if self.std is None:
                std = np.sqrt(2.0 / fan_in)  # He initialization
            else:
                std = self.std               # fixed std, as in ClassificationNet
            self.params['W' + str(i)] = std * np.random.randn(fan_in, fan_out)
            self.params['b' + str(i)] = np.zeros(fan_out)

        # The optimizer expects one gradient entry per parameter.
        self.grads = {name: 0.0 for name in self.params}
        self.reg = {}

    def forward(self, X):
        out = None
        ########################################################################
        # TODO:  Your forward here                                             #
        ########################################################################

        # Layer i (1-based):  out = affine(out, Wi, bi)
        #                     out = activation(out)    except for the last layer
        # The last layer has no activation: its outputs are the logits, which
        # CrossEntropyFromLogits turns into probabilities via softmax.
        self.cache = {}  # everything backward() needs, per layer
        self.reg = {}    # L2 penalty per weight matrix; the Solver adds it to the loss

        out = X.reshape(X.shape[0], -1)  # (N, 32, 32, 3) -> (N, 3072); no-op if already flat
        for i in range(1, self.num_layer + 1):
            W, b = self.params['W' + str(i)], self.params['b' + str(i)]

            out, self.cache['affine' + str(i)] = affine_forward(out, W, b)

            is_last_layer = (i == self.num_layer)
            if not is_last_layer:
                out, self.cache['activation' + str(i)] = self.activation.forward(out)

            # L2 regularization: reg * sum(W^2)  (biases are not regularized)
            self.reg['W' + str(i)] = self.reg_strength * np.sum(W ** 2)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return out

    def backward(self, dy):
        grads = None
        ########################################################################
        # TODO:  Your backward here                                            #
        ########################################################################

        # Walk through the layers in REVERSE order and undo every forward step:
        #   forward  layer i:  affine  -> activation
        #   backward layer i:  activation_backward -> affine_backward
        # dout always holds the gradient of the loss w.r.t. the current output.
        dout = dy  # gradient w.r.t. the logits, from the loss function
        for i in range(self.num_layer, 0, -1):
            is_last_layer = (i == self.num_layer)
            if not is_last_layer:
                dout = self.activation.backward(dout, self.cache['activation' + str(i)])

            dout, dW, db = affine_backward(dout, self.cache['affine' + str(i)])

            # Gradient of the L2 term reg * sum(W^2) w.r.t. W is 2 * reg * W.
            self.grads['W' + str(i)] = dW + 2 * self.reg_strength * self.params['W' + str(i)]
            self.grads['b' + str(i)] = db

        grads = self.grads

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return grads

