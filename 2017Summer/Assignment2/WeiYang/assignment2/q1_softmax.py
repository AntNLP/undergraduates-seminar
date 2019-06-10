import numpy as np
import dynet as dy
from utils.general_utils import test_all_close


def softmax(x):
    ### YOUR CODE HERE
    x_max = dy.max_dim(x, 1)
    x_sub = dy.colwise_add(x, -x_max)
    x_exp = dy.exp(x_sub)
    x_sum = dy.sum_cols(x_exp)
    x_tmp = dy.zeroes(x.dim()[0])
    x_tmp = dy.colwise_add(x_tmp, x_sum)
    out = dy.cdiv(x_exp, x_tmp)
    ### END YOUR CODE
    return out


def cross_entropy_loss(y, yhat):
    ### YOUR CODE HERE
    out = dy.sum_elems(-dy.cmult(y, dy.log(yhat)))
    ### END YOUR CODE
    return out


def test_softmax_basic():
    """
    Some simple tests of softmax to get you started.
    Warning: these are not exhaustive.
    """

    # test1 = softmax(torch.Tensor([[1001, 1002], [3, 4]]))
    # test1 = test1.numpy()
    test1 = softmax(dy.inputTensor([[1001, 1002], [3, 4]]))
    test1 = test1.npvalue();
    test_all_close("Softmax test 1", test1, np.array([[0.26894142,  0.73105858],
                                                      [0.26894142,  0.73105858]]))

    # test2 = softmax(torch.Tensor([[-1001, -1002]]))
    # test2 = test2.numpy()
    test2 = softmax(dy.inputTensor([[-1001, -1002]]))
    test2 = test2.npvalue();
    test_all_close("Softmax test 2", test2, np.array([[0.73105858, 0.26894142]]))

    print "Basic (non-exhaustive) softmax tests pass\n"


def test_cross_entropy_loss_basic():
    """
    Some simple tests of cross_entropy_loss to get you started.
    Warning: these are not exhaustive.
    """
    y = np.array([[0, 1], [1, 0], [1, 0]])
    yhat = np.array([[.5, .5], [.5, .5], [.5, .5]])

    # test1 = cross_entropy_loss(
    #         torch.Tensor([[0, 1], [1, 0], [1, 0]]),
    #        torch.Tensor([[.5, .5], [.5, .5], [.5, .5]]))
    # test1 = np.array(test1)
    test1 = cross_entropy_loss(
            dy.inputTensor([[0, 1], [1, 0], [1, 0]]),
           dy.inputTensor([[.5, .5], [.5, .5], [.5, .5]]))
    test1 = np.array(test1.value())
    expected = -3 * np.log(.5)
    test_all_close("Cross-entropy test 1", test1, expected)

    print "Basic (non-exhaustive) cross-entropy tests pass"

if __name__ == "__main__":
    test_softmax_basic()
    test_cross_entropy_loss_basic()
