# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import time

import numpy as np
import tensorflow as tf
import dynet as dy

from q1_softmax import softmax
from q1_softmax import cross_entropy_loss
from model import Model
from utils.general_utils import get_minibatches


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    n_samples = 1024
    n_features = 100
    n_classes = 5
    batch_size = 64
    n_epochs = 50
    lr = 1e-4


class SoftmaxModel(Model):
    """ a Softmax classifier with cross-entropy loss."""

    def init_trainer(self):
        """Sets up the trainer.

        """
        ### YOUR CODE HERE
        self.sModel = dy.ParameterCollection()
        self.trainer = dy.SimpleSGDTrainer(self.sModel)
        self.trainer.learning_rate = self.config.lr
        ### END YOUR CODE

    def init_parameters(self):
        """Set up parameters

        """
        ### YOUR CODE HERE
        self._pW = self.sModel.add_parameters((self.config.n_features, self.config.n_classes))
        self._pb = self.sModel.add_parameters((self.config.n_classes))
        # associate the parameters with cg Expressions

        ### END YOUR CODE

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """Creates the feed_dict for training the given step.

        If label_batch is None, then no labels are added to feed_dict.

        Hint: The keys for the feed_dict should be the placeholder
                tensors created in add_placeholders.

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        ### YOUR CODE HERE
        self.input = inputs_batch
        self.labels = labels_batch
        ### END YOUR CODE

    def prediction(self):
        """Adds the core transformation for this model which transforms a batch of input
        data into a batch of predictions. In this case, the transformation is a linear layer plus a
        softmax transformation:

        y = softmax(xW + b)

        Args:
            input_data: A tensor of shape (batch_size, n_features).
        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        W = dy.parameter(self._pW)
        b = dy.parameter(self._pb)
        x = dy.inputTensor(self.input)

        z_m = x * W
        z_T = dy.concatenate_cols([z_m[i]+b for i in range(self.config.batch_size)])
        z = dy.transpose(z_T)
        # z = x * W + b

        pred = softmax(z)
        return pred

    def compute_loss(self, pred):
        """Adds cross_entropy_loss ops to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar)
        """
        y = dy.inputTensor(self.labels)
        loss = cross_entropy_loss(y, pred)
        return loss

    def run_epoch(self, inputs, labels):
        """Runs an epoch of training.

        Args:
            inputs: np.ndarray of shape (n_samples, n_features)
            labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """
        config = self.config
        n_minibatches, total_loss = 0, 0
        for input_batch, labels_batch in get_minibatches([inputs, labels], config.batch_size):
            n_minibatches += 1
            dy.renew_cg()
            '''Compute the loss of a batch'''
            # loss = []
            # for i in xrange(config.batch_size):
            #     input_t, labels_t = input_batch[i].reshape(1, config.n_features), labels_batch[i].reshape(1, config.n_classes)
            #     loss_t = self.train_on_batch(input_t, labels_t)
            #     loss.append(loss_t)
            # loss = dy.esum(loss) / config.batch_size
            loss = self.train_on_batch(input_batch, labels_batch) / config.batch_size

            loss.forward()
            loss.backward()
            self.trainer.update()

            total_loss += loss.value()
        return total_loss / n_minibatches

    def fit(self, inputs, labels):
        """Fit model on provided data.

        Args:
            inputs: np.ndarray of shape (n_samples, n_features)
            labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
            losses: list of loss per epoch
        """
        losses = []
        for epoch in range(self.config.n_epochs):
            start_time = time.time()
            average_loss = self.run_epoch(inputs, labels)
            duration = time.time() - start_time
            print 'Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch, average_loss, duration)
            losses.append(average_loss)
        return losses

    def __init__(self, config):
        """Initializes the model.

        Args:
            config: A model configuration object of type Config
        """
        self.config = config
        self.build()


def test_softmax_model():
    """Train softmax model for a number of steps."""
    config = Config()

    # Generate random data to train the model on
    np.random.seed(1234)
    inputs = np.random.rand(config.n_samples, config.n_features)
    labels = np.zeros((config.n_samples, config.n_classes), dtype=np.int32)
    labels[:, 1] = 1
    # for i in xrange(config.n_samples):
    #     labels[i, i%config.n_classes] = 1

    model = SoftmaxModel(config)
    losses = model.fit(inputs, labels)

    # If Ops are implemented correctly, the average loss should fall close to zero
    # rapidly.
    assert losses[-1] < .5
    print "Basic (non-exhaustive) classifier tests pass"

if __name__ == "__main__":
    test_softmax_model()
