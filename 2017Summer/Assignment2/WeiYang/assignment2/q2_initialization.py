import numpy as np
import dynet as dy


def xavier_weight_init():
    def _xavier_initializer(shape, **kwargs):
        ### YOUR CODE HERE
        epsilon = np.sqrt(6 / np.sum(shape))
        out = dy.random_uniform(dim=shape, left=-epsilon, right=epsilon)
        ### END YOUR CODE
        return out
    return _xavier_initializer


def test_initialization_basic():
    print "Running basic tests..."
    xavier_initializer = xavier_weight_init()
    shape = (1,)
    xavier_mat = xavier_initializer(shape)
    assert xavier_mat.dim()[0] == shape

    shape = (1, 2, 3)
    xavier_mat = xavier_initializer(shape)
    assert xavier_mat.dim()[0] == shape
    print "Basic (non-exhaustive) Xavier initialization tests pass"


if __name__ == "__main__":
    test_initialization_basic()
