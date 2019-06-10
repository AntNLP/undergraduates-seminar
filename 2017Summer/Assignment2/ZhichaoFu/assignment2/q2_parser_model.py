# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os
import time
import tensorflow as tf
import dynet as dy
import numpy as np
import cPickle

from model import Model
from q2_initialization import xavier_weight_init
from utils.general_utils import Progbar
from utils.parser_utils import minibatches, load_and_preprocess_data



class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    n_features = 36
    n_classes = 3
    dropout = 0.5
    embed_size = 50
    hidden_size = 200
    batch_size = 2048
    n_epochs = 10
    lr = 0.001


class ParserModel(Model):
    """
    Implements a feedforward neural network with an embedding layer and single hidden layer.
    This network will predict which transition should be applied to a given partial parse
    configuration.
    """

    def init_trainer(self):
        """Sets up the trainer.

        """
        ### YOUR CODE HERE
        self.M = dy.ParameterCollection()
        self.trainer = dy.AdamTrainer(self.M)
        self.trainer.learning_rate = self.config.lr
        ### END YOUR CODE

    def init_parameters(self):
        """Set up parameters

        Use the initializer from q2_initialization.py to initialize W and U (you can initialize b1
        and b2 with zeros)
        """
        ### YOUR CODE HERE
        zeroInit = dy.ConstInitializer(0.0)
        xavier = xavier_weight_init()

        self._pW = self.M.add_parameters((self.config.n_features * self.config.embed_size, self.config.hidden_size))
        self._pB1 = self.M.add_parameters((1, self.config.hidden_size), init=zeroInit)
        self._pU = self.M.add_parameters((self.config.hidden_size, self.config.n_classes))
        self._pB2 = self.M.add_parameters((1, self.config.n_classes), init=zeroInit)

        self.word_dict = self.M.lookup_parameters_from_numpy(self.pretrained_embeddings)
        ### END YOUR CODE

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for the dependency parser.

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        ### YOUR CODE HERE
        self.input = inputs_batch
        self.labels = labels_batch
        self.dropout = dropout
        ### END YOUR CODE

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:
            - Creates an embedding tensor and initializes it with self.pretrained_embeddings.
            - Uses the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, n_features, embedding_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, n_features * embedding_size).

        Returns:
            embeddings: dy.Tensor of shape (None, n_features*embed_size)
        """
        ### YOUR CODE HERE
        embeddings = dy.concatenate([self.word_dict.batch(x) for x in np.transpose(self.input)])
        embeddings = dy.transpose(embeddings)
        ### END YOUR CODE
        return embeddings

    def prediction(self, dropout=False):
        """Adds the 1-hidden-layer NN:
            h = Relu(xW + b1)
            h_drop = Dropout(h, dropout_rate)
            pred = h_dropU + b2


        Hint: Here are the dimensions of the various variables you will need to create
                    W:  (n_features*embed_size, hidden_size)
                    b1: (hidden_size,)
                    U:  (hidden_size, n_classes)
                    b2: (n_classes)

        Returns:
            pred: dy.Tensor of shape (batch_size, n_classes)
        """

        x = self.add_embedding()
        ### YOUR CODE HERE
        W = dy.parameter(self._pW)
        U = dy.parameter(self._pU)
        b1 = dy.parameter(self._pB1)
        b2 = dy.parameter(self._pB2)

        z1 = x * W + b1
        h = dy.rectify(z1)
        h_drop = dy.dropout(h, self.dropout) if dropout else h

        z2 = h_drop * U + b2

        pred = dy.softmax(dy.reshape(z2, (self.config.n_classes,)))
        ### END YOUR CODE
        return pred

    def compute_loss(self, pred):
        """Adds Ops for the loss function to the computational graph.
        In this case we are using cross entropy loss.
        The loss should be averaged over all examples in the current minibatch.

        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        ### YOUR CODE HERE
        y = dy.inputTensor(np.transpose(self.labels), batched=True)

        losses = dy.binary_log_loss(pred, y)
        loss = dy.sum_batches(losses) / self.config.batch_size
        ### END YOUR CODE
        return loss


    def train_on_batch(self, inputs_batch, labels_batch):
        self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)

        pred = self.prediction(dropout=True)

        loss = self.compute_loss(pred)

        return loss

    def predict_on_batch(self, inputs_batch):
        self.create_feed_dict(inputs_batch)

        pred_dy = self.prediction()

        pred = np.transpose(pred_dy.npvalue())

        return pred

    def run_epoch(self, parser, train_examples, dev_set):
        prog = Progbar(target=1 + len(train_examples) / self.config.batch_size)
        for i, (train_x, train_y) in enumerate(minibatches(train_examples, self.config.batch_size)):
            dy.renew_cg()
            loss = self.train_on_batch(train_x, train_y)
            loss.forward()
            loss.backward()
            self.trainer.update()

            prog.update(i + 1, [("train loss", loss.value())])

        print "Evaluating on dev set",
        dev_UAS, _ = parser.parse(dev_set)
        print "- dev UAS: {:.2f}".format(dev_UAS * 100.0)
        return dev_UAS

    def fit(self, saver, parser, train_examples, dev_set):
        best_dev_UAS = 0
        for epoch in range(self.config.n_epochs):
            print "Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs)
            dev_UAS = self.run_epoch(parser, train_examples, dev_set)
            if dev_UAS > best_dev_UAS:
                best_dev_UAS = dev_UAS
                if saver:
                    print "New best dev UAS! Saving model in ./data/weights/parser.weights"
                    self.M.save_all('./data/weights/parser.weights')
            print

    def __init__(self, config, pretrained_embeddings):
        self.pretrained_embeddings = pretrained_embeddings
        self.config = config
        self.build()


def main(debug=True):
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="
    config = Config()
    parser, embeddings, train_examples, dev_set, test_set = load_and_preprocess_data(debug)
    if not os.path.exists('./data/weights/'):
        os.makedirs('./data/weights/')

    print "Building model...",
    start = time.time()
    model = ParserModel(config, embeddings)
    parser.model = model
    print "took {:.2f} seconds\n".format(time.time() - start)

    saver = None if debug else True

    print 80 * "="
    print "TRAINING"
    print 80 * "="
    model.fit(saver, parser, train_examples, dev_set)

    if False:
        print 80 * "="
        print "TESTING"
        print 80 * "="
        print "Restoring the best model weights found on the dev set"
        saver.restore(session, './data/weights/parser.weights')
        print "Final evaluation on test set",
        UAS, dependencies = parser.parse(test_set)
        print "- test UAS: {:.2f}".format(UAS * 100.0)
        print "Writing predictions"
        with open('q2_test.predicted.pkl', 'w') as f:
            cPickle.dump(dependencies, f, -1)
        print "Done!"

if __name__ == '__main__':
    main()


