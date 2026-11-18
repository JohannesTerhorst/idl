
import os
import pickle
import numpy as np

from exercise_code.networks.base_networks import Network


class Classifier(Network):
    """
    Classifier of the form y = sigmoid(X * W)
    """

    def __init__(self, num_features=2):
        super().__init__("classifier")

        self.num_features = num_features 
        self.W = None
        self.cache = None
        
        self.initialize_weights()

    def initialize_weights(self, weights=None):
        """
        Initialize the weight matrix W

        :param weights: optional weights for initialization
        """
        if weights is not None:
            assert weights.shape == (self.num_features + 1, 1), \
                "weights for initialization are not in the correct shape (num_features + 1, 1)"
            self.W = weights
        else:
            self.W = 0.001 * np.random.randn(self.num_features + 1, 1)

    def forward(self, X):
        """
        Performs the forward pass of the model.

        :param X: N x D array of training data. Each row is a D-dimensional point. 
            Note that it is changed to N x (D + 1) to include the bias term.
        :return: Predicted logits for the data in X, shape N x 1
                 1-dimensional array of length N with classification scores.
                 
        Note: This simple neural-network contains TWO consecutive layers: 
        A fully-connected layer and a sigmoid layer.
        """
        assert self.W is not None, "weight matrix W is not initialized"
        # add a column of 1s to the data for the bias term
        batch_size, _ = X.shape
        X = np.concatenate((X, np.ones((batch_size, 1))), axis=1)
        z = None


        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass and return the output of the model. Note  #
        # that you need to implement the function self.sigmoid() for that.     #
        # Also, save in self.cache an array of all the relevant variables that #
        # you will need to use in the backward() function. E.g.: (X, ...)      #
        #                                                                      #
        # Hint:                                                                #
        #       The function is made up of TWO functions: Affine and sigmoid.  #
        #       The sigmoid is applied to the result of the affine step.       #
        ########################################################################
        # --- 1. EXPLANATION ---------------------------------------------------
        # Logistic regression = linear (affine) layer followed by a sigmoid:
        #       s = X @ W            (N x (D+1)) @ ((D+1) x 1) -> N x 1
        #       z = sigmoid(s)       probability that the label is 1
        # The bias is already handled: X got an extra column of ones above,
        # so the last row of W acts as the bias b.
        # For the backward pass we need X (for dW = X^T ...) and z (for the
        # sigmoid derivative z(1-z)), so we store them in the cache.
        #
        # --- 2. PSEUDO CODE ---------------------------------------------------
        #   s          = X @ self.W         (or np.dot(X, self.W))
        #   z          = self.sigmoid(s)
        #   self.cache = (X, z)             (optionally s as well)
        #
        # --- 3. WATCH OUT / OPTIMIZATION -------------------------------------
        # - Use matrix multiplication (@ / np.dot), NOT element-wise '*'.
        # - Cache the X WITH the ones column (the one after concatenate),
        #   otherwise dW will have shape D x 1 instead of (D+1) x 1.
        # - Storing z means the backward pass does not have to recompute the
        #   sigmoid (cheap here, but a key idea for bigger networks).
        ########################################################################

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return z 

    def backward(self, dout):
        """
        Performs the backward pass of the model.

        :param dout: N x M array. Upsteam derivative. It is as the same shape of the forward() output.
                     If the output of forward() is z, then it is dL/dz, where L is the loss function.
        :return: dW --> Gradient of the weight matrix, w.r.t the upstream gradient 'dout'. (dL/dw)
        
        Note: Pay attention to the order in which we calculate the derivatives. It is the opposite of the forward pass!
        """
        assert self.cache is not None, ("Run a forward pass before the backward pass. Also, don't forget to store the"
                                        " relevant variables such as in 'self.cache = (X, y, ...)")
        dW = None

        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass. Return the gradient w.r.t W --> dW.     #
        # Make sure you've stored ALL needed variables in self.cache. (!!!!)   #
        #                                                                      #
        # Hint 1: It is recommended to follow the TUM article (Section 3) on   #
        # calculating the chain-rule, while dealing with matrix notations:     #
        # https://bit.ly/tum-article                                           #
        #                                                                      #   
        # Hint 2: Remember that the derivative of sigmoid(x) is independent of #
        # x, and could be calculated with the result from the forward pass.    #
        #                                                                      #
        # Hint 3: The argument "dout" stands for the upstream gradeint to this #
        # layer.                                                               #
        ########################################################################
        # --- 1. EXPLANATION ---------------------------------------------------
        # Chain rule, in REVERSE order of the forward pass (s = XW, z = σ(s)):
        #
        #   dL/dz = dout                         (given by the loss backward)
        #   dL/ds = dL/dz * dz/ds = dout * z * (1 - z)      (element-wise)
        #           because σ'(s) = σ(s)(1 - σ(s)) = z(1 - z)
        #   dL/dW = X^T @ dL/ds                  (matrix product)
        #
        # Why X^T? Each weight W_j influences every sample through X[i, j],
        # so dL/dW_j = sum_i X[i, j] * dL/ds_i, which is exactly X^T @ dL/ds.
        #
        # --- 2. PSEUDO CODE ---------------------------------------------------
        #   X, z = self.cache
        #   ds   = dout * z * (1 - z)           # N x 1
        #   dW   = X.T @ ds                     # (D+1) x N @ N x 1 -> (D+1) x 1
        #
        # --- 3. WATCH OUT / OPTIMIZATION -------------------------------------
        # - Shape check: dW must have the same shape as self.W ((D+1) x 1).
        #   If dout comes in as (N,), reshape it to (N, 1) first, otherwise
        #   dout * z broadcasts to N x N.
        # - Do NOT divide by N here: the 1/N is already inside dout
        #   (from the loss backward). Dividing twice shrinks the gradient.
        # - '*' for the element-wise sigmoid part, '@' for the X^T product.
        # - Use the cached z, don't call sigmoid again.
        ########################################################################

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return dW # (dL/dW)

    def sigmoid(self, x):
        """
        Computes the ouput of the sigmoid function.

        :param x: input of the sigmoid, np.array of any shape
        :return: output of the sigmoid with same shape as input vector x
        """
        out = None

        ########################################################################
        # TODO:                                                                #
        # Implement the sigmoid function over the input x. Return "out".       #
        # Note: The sigmoid() function operates element-wise.                  #
        ########################################################################
        # --- 1. EXPLANATION ---------------------------------------------------
        # The sigmoid squashes any real number into (0, 1), so the output can
        # be read as a probability:
        #       σ(x) = 1 / (1 + e^(-x))
        # σ(0) = 0.5, σ(large +) -> 1, σ(large -) -> 0.
        # Its derivative is σ(x) * (1 - σ(x)) (used in backward()).
        #
        # --- 2. PSEUDO CODE ---------------------------------------------------
        #   out = 1 / (1 + np.exp(-x))
        #
        # --- 3. WATCH OUT / OPTIMIZATION -------------------------------------
        # - Use np.exp (works element-wise on any array shape), not math.exp.
        # - For very negative x, np.exp(-x) overflows -> RuntimeWarning
        #   (result still becomes 0, so it usually works). A stable version:
        #     x >= 0:  1 / (1 + exp(-x))
        #     x <  0:  exp(x) / (1 + exp(x))
        #   e.g. via np.where, or np.clip(x, -500, 500) before exp.
        ########################################################################

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return out

    def save_model(self):
        directory = 'models'
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(
            model,
            open(
                directory +
                '/' +
                self.model_name +
                '.p',
                'wb'))
