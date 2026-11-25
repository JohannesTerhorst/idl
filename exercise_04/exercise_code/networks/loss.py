
import os
import pickle
import numpy as np
# from exercise_code.networks.linear_model import *


class Loss(object):
    def __init__(self):
        self.grad_history = []

    def forward(self, y_out, y_truth, individual_losses=False):
        return NotImplementedError

    def backward(self, y_out, y_truth, upstream_grad=1.):
        return NotImplementedError

    def __call__(self, y_out, y_truth, individual_losses=False):
        loss = self.forward(y_out, y_truth, individual_losses)
        return loss


class L1(Loss):

    def forward(self, y_out, y_truth, individual_losses=False):
        """
        Performs the forward pass of the L1 loss function.

        :param y_out: [N, ] array predicted value of your model.
        :y_truth: [N, ] array ground truth value of your training set.
        :return: 
            - individual_losses=False --> A single scalar, which is the mean of the binary cross entropy loss 
                for each sample of your training set.
            - individual_losses=True  --> [N, ] array of binary cross entropy loss for each sample of your training set.
        """
        result = None

        result = np.abs(y_out - y_truth)
        
        if individual_losses:
            return result
        return np.mean(result)

    def backward(self, y_out, y_truth):
        """
        Performs the backward pass of the L1 loss function.

        :param y_out: [N, ] array predicted value of your model.
        :y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of L1 loss gradients w.r.t y_out for each sample of your training set.
        """
        gradient = None
        gradient = y_out - y_truth

        zero_loc = np.where(gradient == 0)
        negative_loc = np.where(gradient < 0)
        positive_loc = np.where(gradient > 0)

        gradient[zero_loc] = 0
        gradient[positive_loc] = 1
        gradient[negative_loc] = -1
    
        return gradient / len(y_out)


class MSE(Loss):

    def forward(self, y_out, y_truth, individual_losses=False):
        """
        Performs the forward pass of the MSE loss function.

        :param y_out: [N, ] array predicted value of your model.
        :y_truth: [N, ] array ground truth value of your training set.
        :return: 
            - individual_losses=False --> A single scalar, which is the mean of the binary cross entropy loss 
                for each sample of your training set.
            - individual_losses=True  --> [N, ] array of binary cross entropy loss for each sample of your training set.
        """
        result = None
        
        result = (y_out - y_truth) ** 2
        
        if individual_losses:
            return result
        return np.mean(result)

    def backward(self, y_out, y_truth):
        """
        Performs the backward pass of the MSE loss function.

        :param y_out: [N, ] array predicted value of your model.
        :y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of MSE loss gradients w.r.t y_out for each sample of your training set.
        """
        gradient = None
        gradient = 2 * (y_out - y_truth) / len(y_out)
        return gradient


class BCE(Loss):

    def forward(self, y_out, y_truth, individual_losses=False):
        """
        Performs the forward pass of the binary cross entropy loss function.

        :param y_out: [N, ] array predicted value of your model (the Logits).
        :y_truth: [N, ] array ground truth value of your training set.
        :return: 
            - individual_losses=False --> A single scalar, which is the mean of the binary cross entropy loss 
                for each sample of your training set.
            - individual_losses=True  --> [N, ] array of binary cross entropy loss values for each sample of your batch.
        """

        y_out   = np.clip(y_out.reshape(-1), 1e-12, 1 - 1e-12)
        y_truth = y_truth.reshape(-1)
        result  = -(y_truth * np.log(y_out) + (1 - y_truth) * np.log(1 - y_out))
        
        if individual_losses:
            return result   # return a list of loss values, without taking the mean.
        result =  np.mean(result)

        return result

    def backward(self, y_out, y_truth):
        """
        Performs the backward pass of the loss function.

        :param y_out: [N, ] array predicted value of your model.
        :y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of binary cross entropy loss gradients w.r.t y_out 
                for each sample of your training set.
        """
        gradient = None

        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass. Return the gradient w.r.t to the input  #
        # to the loss function, y_out.                                         #
        #                                                                      #
        # Hint:                                                                #
        #   Don't forget to divide by N, which is the number of samples in     #
        #   the batch. It is crucial for the magnitude of the gradient.        #
        ########################################################################
        # --- 1. EXPLANATION ---------------------------------------------------
        # We need dL/dp for every sample, where L = mean_i(L_i) and p = y_out.
        # Differentiate L_i = -[ y log(p) + (1-y) log(1-p) ] w.r.t. p:
        #
        #       dL_i/dp_i = -( y_i / p_i  -  (1 - y_i) / (1 - p_i) )
        #
        # Because the forward pass takes the MEAN over N samples, every
        # per-sample gradient is additionally multiplied by 1/N.
        #
        # --- 2. PSEUDO CODE ---------------------------------------------------
        #   N        = number of samples (len(y_out) / y_out.shape[0])
        #   gradient = -( y_truth / y_out - (1 - y_truth) / (1 - y_out) )
        #   gradient = gradient / N
        #
        # --- 3. WATCH OUT / OPTIMIZATION -------------------------------------
        # - Division by zero if p is exactly 0 or 1 -> clip p like in forward.
        # - Same shape trap as in forward: gradient must have the SAME shape
        #   as y_out (it is passed as 'dout' to Classifier.backward()).
        # - Nice fact: chaining this with the sigmoid derivative p(1-p) gives
        #   dL/ds = (p - y) / N. Frameworks fuse sigmoid+BCE for exactly this
        #   reason (cheaper and numerically stable). Here, keep them separate
        #   because the classifier multiplies by p(1-p) itself.
        # - Sanity check: compare with a numerical gradient
        #   (L(p + h) - L(p - h)) / 2h for a tiny h.
        ########################################################################

        gradient = -(y_truth / y_out - (1 - y_truth) / (1 - y_out)) / len(y_out)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return gradient


