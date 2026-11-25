# Naive Optimizer using full batch gradient descent
import os
import pickle
import numpy as np
from exercise_code.networks.linear_model import *


class Optimizer(object):
    def __init__(self, model, learning_rate=5e-5):
        self.model = model
        self.lr = learning_rate

    def step(self, dw):
        """
        A vanilla gradient descent step.
        
        :param dw: [D+1,1] array gradient of loss w.r.t weights of your linear model
        :return weight: [D+1,1] updated weight after one step of gradient descent.
        """
        weight = self.model.W

        ########################################################################
        # TODO:                                                                #
        # Implement the gradient descent step over the weight, using the       #
        # learning rate.                                                       #
        ########################################################################
        # --- 1. EXPLANATION ---------------------------------------------------
        # Gradient descent: the gradient dw points in the direction where the
        # loss INCREASES fastest, so we take a small step in the opposite
        # direction. The learning rate (self.lr) controls the step size:
        #       W_new = W_old - lr * dL/dW
        #
        # --- 2. PSEUDO CODE ---------------------------------------------------
        #   weight = weight - self.lr * dw
        #
        # --- 3. WATCH OUT / OPTIMIZATION -------------------------------------
        # - Minus sign, not plus (plus = gradient ASCENT, the loss explodes).
        # - Prefer 'weight = weight - ...' over in-place 'weight -= ...':
        #   in-place edits self.model.W directly, which can surprise you if
        #   someone else holds a reference to the same array.
        # - lr too large -> loss oscillates / diverges; too small -> very
        #   slow training. Watch the loss curve in the notebook.
        # - Improvements beyond this exercise: mini-batch SGD, momentum,
        #   Adam (adaptive per-parameter step sizes), learning-rate decay.
        ########################################################################

        weight = weight - self.lr * dw

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        self.model.W = weight
