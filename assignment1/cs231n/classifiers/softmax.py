import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  classes = W.shape[1]
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
      scores = X[i].dot(W)
        
      max_score = np.amax(scores, axis=-1)
      scores -= max_score
      scores = np.exp(scores)
      sum_score = np.sum(scores, axis=-1)
      scores /= sum_score
    
      loss += -1 * np.log(scores[y[i]])
      
      for c in xrange(classes):
          if c == y[i]:
              dW[:, c] += (scores[c] - 1) * X[i]
          else:
              dW[:, c] += scores[c] * X[i]
                
  loss /= num_train
  loss += reg * np.sum(W*W)
    
  dW /= num_train
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores -= np.amax(scores, axis=-1, keepdims=True)
  scores = np.exp(scores)
  sum_scores = np.sum(scores, axis=-1, keepdims=True)
  scores /= sum_scores
  
  for i in xrange(num_train):
      loss += -1 * np.log(scores[i, y[i]])
      scores[i, y[i]] -= 1
        
  dW = (X.T).dot(scores)
  
  loss /= num_train
  loss += reg*np.sum(W*W)
  
  dW /= num_train
  dW +=reg * 2 * W  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

