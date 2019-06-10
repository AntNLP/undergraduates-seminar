# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import dynet as dy

class Model(object):
    """Abstracts a Tensorflow graph for a learning task.

    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs. Each algorithm you will construct in this homework will
    inherit from a Model object.
    """

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """Creates the feed_dict for one step of training.

        If labels_batch is None, then no labels are added to feed_dict.

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def init_parameters(self):
        """Initialize parameters for the Dynet model

        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def init_trainer(self):
        """Sets up the trainer.
        """

        raise NotImplementedError("Each Model must re-implement this method.")

    def train_on_batch(self, inputs_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            input_batch: np.ndarray of shape (n_samples, n_features)
            labels_batch: np.ndarray of shape (n_samples, n_classes)
        Returns:
            loss: loss over the batch (a scalar)
        """

        self.create_feed_dict(inputs_batch, labels_batch=labels_batch)

        pred = self.prediction()

        loss = self.compute_loss(pred)

        return loss

    def predict_on_batch(self, inputs_batch):
        """Make predictions for the provided batch of data

        Args:
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        self.create_feed_dict(inputs_batch)

        pred = self.prediction()

        return pred

    def build(self):
        self.init_trainer()
        self.init_parameters()
