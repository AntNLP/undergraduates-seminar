# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
#import tensorflow as tf
import dynet as dy
from utils.general_utils import test_all_close

def softmax(x):
    """
    Compute the softmax function in tensorflow.

    You might find the tensorflow functions tf.exp, tf.reduce_max,
    tf.reduce_sum, tf.expand_dims useful. (Many solutions are possible, so you may
    not need to use all of these functions). Recall also that many common
    tensorflow operations are sugared (e.g. x * y does a tensor multiplication
    if x and y are both tensors). Make sure to implement the numerical stability
    fixes as in the previous homework!

    Args:
        x:   tf.Tensor with shape (n_samples, n_features). Note feature vectors are
                  represented by row-vectors. (For simplicity, no need to handle 1-d
                  input as in the previous homework)
    Returns:
        out: tf.Tensor with shape (n_sample, n_features). You need to construct this
                  tensor in this problem.
    """

    ### YOUR CODE HERE
    x_max = dy.max_dim(x, 1)
    x_sub = dy.colwise_add(x, -x_max)
    x_exp = dy.exp(x_sub)
    sum_exp = dy.colwise_add(dy.zeroes(x.dim()[0]), dy.sum_cols(x_exp))

    out = dy.cdiv(x_exp, sum_exp)
    ### END YOUR CODE

    return out


def cross_entropy_loss(y, yhat):
    """
    Compute the cross entropy loss in tensorflow.
    The loss should be summed over the current minibatch.

    y is a one-hot tensor of shape (n_samples, n_classes) and yhat is a tensor
    of shape (n_samples, n_classes). y should be of dtype tf.int32, and yhat should
    be of dtype tf.float32.

    The functions tf.to_float, tf.reduce_sum, and tf.log might prove useful. (Many
    solutions are possible, so you may not need to use all of these functions).

    Note: You are NOT allowed to use the tensorflow built-in cross-entropy
                functions.

    Args:
        y:    tf.Tensor with shape (n_samples, n_classes). One-hot encoded.
        yhat: tf.Tensorwith shape (n_sample, n_classes). Each row encodes a
                    probability distribution and should sum to 1.
    Returns:
        out:  tf.Tensor with shape (1,) (Scalar output). You need to construct this
                    tensor in the problem.
    """

    ### YOUR CODE HERE
    l_yhat = dy.log(yhat)
    product = dy.cmult(y, l_yhat)
    out = (-dy.sum_elems(product))
    ### END YOUR CODE

    return out


def test_softmax_basic():
    """
    Some simple tests of softmax to get you started.
    Warning: these are not exhaustive.
    """
    dy.renew_cg()
    test1 = softmax(dy.inputTensor(np.array([[1001, 1002], [3, 4]])))
    test_all_close("Softmax test 1", test1.value(), np.array([[0.26894142,  0.73105858],
                                                      [0.26894142,  0.73105858]]))
    dy.renew_cg()
    test2 = softmax(dy.inputTensor(np.array([[-1001, -1002]])))
    test_all_close("Softmax test 2", test2.value(), np.array([[0.73105858, 0.26894142]]))

    print "Basic (non-exhaustive) softmax tests pass\n"


def test_cross_entropy_loss_basic():
    """
    Some simple tests of cross_entropy_loss to get you started.
    Warning: these are not exhaustive.
    """
    y = np.array([[0, 1], [1, 0], [1, 0]])
    yhat = np.array([[.5, .5], [.5, .5], [.5, .5]])

    test1 = cross_entropy_loss(dy.inputTensor(y), dy.inputTensor(yhat))

    expected = -3 * np.log(.5)
    test_all_close("Cross-entropy test 1", np.array(test1.value()), expected)

    print "Basic (non-exhaustive) cross-entropy tests pass"

if __name__ == "__main__":
    test_softmax_basic()
    test_cross_entropy_loss_basic()
