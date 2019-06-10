import time

import numpy as np
import dynet as dy

from q1_softmax import softmax
from q1_softmax import cross_entropy_loss
from model import Model
from utils.general_utils import get_minibatches


class Config(object):
    n_samples = 1024
    n_features = 100
    n_classes = 5
    batch_size = 64
    n_epochs = 50
    lr = 1e-4


class SoftmaxModel(Model):
    def init_trainer(self):
        ### YOUR CODE HERE
        self.sModel = dy.ParameterCollection()
        self.trainer = dy.SimpleSGDTrainer(self.sModel)
        self.trainer.learning_rate = self.config.lr
        ### END YOUR CODE

    def init_parameters(self):
        ### YOUR CODE HERE
        self._pW = self.sModel.add_parameters((self.config.n_features, self.config.n_classes))
        self._pb = self.sModel.add_parameters((self.config.n_classes))
        ### END YOUR CODE

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        ### YOUR CODE HERE
        self.input = inputs_batch
        self.labels = labels_batch
        ### END YOUR CODE

    def prediction(self):
        W = dy.parameter(self._pW)
        b = dy.parameter(self._pb)
        x = dy.inputTensor(self.input)
        z_m = x * W
        z_T = dy.concatenate_cols([z_m[i]+b for i in range(self.config.batch_size)])
        z = dy.transpose(z_T)
        pred = softmax(z)
        return pred

    def compute_loss(self, pred):
        y = dy.inputTensor(self.labels)
        loss = cross_entropy_loss(y, pred)
        return loss

    def run_epoch(self, inputs, labels):
        config = self.config
        n_minibatches, total_loss = 0, 0
        for input_batch, labels_batch in get_minibatches([inputs, labels], config.batch_size):
            n_minibatches += 1
            dy.renew_cg()
            loss = self.train_on_batch(input_batch, labels_batch) / config.batch_size

            loss.forward()
            loss.backward()
            self.trainer.update()

            total_loss += loss.value()
        return total_loss / n_minibatches

    def fit(self, inputs, labels):
        losses = []
        for epoch in range(self.config.n_epochs):
            start_time = time.time()
            average_loss = self.run_epoch(inputs, labels)
            duration = time.time() - start_time
            print 'Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch, average_loss, duration)
            losses.append(average_loss)
        return losses

    def __init__(self, config):
        self.config = config
        self.build()


def test_softmax_model():
    config = Config()
    np.random.seed(1234)
    inputs = np.random.rand(config.n_samples, config.n_features)
    labels = np.zeros((config.n_samples, config.n_classes), dtype=np.int32)
    labels[:, 1] = 1
    # for i in xrange(config.n_samples):
    #     labels[i, i%config.n_classes] = 1

    model = SoftmaxModel(config)
    losses = model.fit(inputs, labels)
    assert losses[-1] < .5
    print "Basic (non-exhaustive) classifier tests pass"

if __name__ == "__main__":
    test_softmax_model()
